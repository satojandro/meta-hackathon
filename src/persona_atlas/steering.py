"""Steering: push one MLP neuron (or a random control direction) while the model runs.

Where we push: the *input* to `mlp.down_proj`, i.e. the vector of neuron activations `silu(gate·x) * (up·x)`.
Adding `a` to entry n there makes the MLP write an extra `a * W_down[:, n]`. For Llama that lands straight in the
residual stream. Glimmer applies a norm after the MLP, so "add to the MLP output" would be squashed by that norm;
pushing the activation instead goes through the same path as the neuron's natural signal, so one hook works for both.

Dose units: a dose of 1.0 means the extra MLP output has the size of the layer's median residual norm (see
calibrate.py). Coherent text tends to sit around 0.5-0.8; 1.5+ is usually gibberish.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import torch


def get_layers(model):
    """The list of transformer blocks. Most HF decoder models keep them at model.model.layers."""
    for path in ("model.layers", "model.language_model.layers", "transformer.h"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        return obj
    raise ValueError(f"can't find the transformer layers of {type(model).__name__}")


def down_proj(model, layer: int) -> torch.nn.Linear:
    return get_layers(model)[layer].mlp.down_proj


@dataclass
class Steer:
    """One push: add `vector` (length = number of MLP neurons) to the down_proj input of `layer`, at every token."""

    layer: int
    vector: torch.Tensor
    label: str = ""


def neuron_steer(model, layer: int, neuron: int, dose: float, resid_norm: float) -> Steer:
    """Push a single neuron. Its activation is shifted so the MLP output moves by `dose * resid_norm`."""
    w = down_proj(model, layer).weight  # shape [d_model, d_mlp]; column n is neuron n's output direction
    col_norm = w[:, neuron].float().norm().item()
    vec = torch.zeros(w.shape[1])
    vec[neuron] = dose * resid_norm / col_norm
    return Steer(layer, vec, f"L{layer} N{neuron} {dose:+.2f}")


def random_steer(model, layer: int, dose: float, resid_norm: float, seed: int = 0) -> Steer:
    """Control: a random mix of all neurons, scaled so the MLP output moves by the same amount as a neuron steer.

    If a "persona" effect also shows up for random pushes of equal size, the neuron isn't special.
    """
    w = down_proj(model, layer).weight.float()
    r = torch.randn(w.shape[1], generator=torch.Generator().manual_seed(seed))
    out_norm = (w @ r.to(w.device)).norm().item()
    return Steer(layer, r * (dose * resid_norm / out_norm), f"L{layer} random#{seed} {dose:+.2f}")


@contextmanager
def steering(model, steers: Steer | list[Steer]):
    """Apply steers inside a `with` block; hooks are always removed afterwards, even on error.

    with steering(model, neuron_steer(model, 5, 1234, 0.8, norms[5])):
        print(chat(model, tok, "Who are you?"))
    """
    if isinstance(steers, Steer):
        steers = [steers]
    handles = []
    try:
        for s in steers:
            proj = down_proj(model, s.layer)
            vec = s.vector.to(device=proj.weight.device, dtype=proj.weight.dtype)

            def pre_hook(module, args, vec=vec):
                return (args[0] + vec, *args[1:])

            handles.append(proj.register_forward_pre_hook(pre_hook))
        yield
    finally:
        for h in handles:
            h.remove()
