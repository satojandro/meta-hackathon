"""Does the steering pipeline work on *this* machine? One command, no cloud.

    uv run python scripts/smoke.py                      # model from .env / MODEL_ID, else Llama 3.2 1B
    uv run python scripts/smoke.py --layer 8 --neurons 5

Loads the model, prints the unsteered answer to "Who are you?", then pushes a few real MLP neurons at
+-0.5 / +-0.8 (a coherent dose in units of the layer's median residual norm) alongside size-matched
random controls. Nothing here is a finding: it is a plumbing check that the hooks, the dose maths and
the chat template all line up on a real model before we trust any experiment.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from persona_atlas import chat, load_model, neuron_steer, random_steer, residual_norms, steering
from persona_atlas.steering import down_proj, get_layers

PROMPT = "Who are you?"


def one_line(text: str, width: int = 100) -> str:
    """Collapse whitespace so each result stays on one readable line."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=None, help="layer to steer (default: middle)")
    ap.add_argument("--neurons", type=int, default=3, help="how many neurons to try")
    ap.add_argument("--doses", type=float, nargs="+", default=[-0.8, 0.8])
    ap.add_argument("--max-new-tokens", type=int, default=40)
    args = ap.parse_args()

    model, tok = load_model()
    n_layers = len(get_layers(model))
    d_mlp = down_proj(model, 0).weight.shape[1]
    print(f"model {getattr(model.config, 'name_or_path', None) or model.config.model_type}")
    print(f"device {model.device}  dtype {next(model.parameters()).dtype}  layers {n_layers}  d_mlp {d_mlp}")

    base = chat(model, tok, PROMPT, max_new_tokens=args.max_new_tokens)
    print(f"\nbaseline  {one_line(base)!r}")

    norms = residual_norms(model, tok)
    layer = n_layers // 2 if args.layer is None else args.layer
    print(f"\nresidual median norm  L0 {norms[0]:.1f}  L{layer} {norms[layer]:.1f}  L{n_layers - 1} {norms[-1]:.1f}")

    # Spread the neurons across the layer instead of the first few, and keep them in range.
    step = max(1, d_mlp // (args.neurons + 1))
    for i, neuron in enumerate(range(step, d_mlp, step)[: args.neurons]):
        for dose in args.doses:
            with steering(model, neuron_steer(model, layer, neuron, dose, norms[layer])):
                out = chat(model, tok, PROMPT, max_new_tokens=args.max_new_tokens)
            print(f"L{layer} N{neuron} {dose:+.1f}  {one_line(out)!r}")

    # Controls: random directions of exactly the same output size. A real persona neuron should look
    # different from these; if the controls produce the same kind of text, nothing was found.
    for seed in range(len(args.doses)):
        with steering(model, random_steer(model, layer, abs(args.doses[0]), norms[layer], seed=seed)):
            out = chat(model, tok, PROMPT, max_new_tokens=args.max_new_tokens)
        print(f"L{layer} random#{seed} {abs(args.doses[0]):+.1f}  {one_line(out)!r}")

    # Clean-up check: if a hook leaked, the "baseline" every later experiment compares against is steered.
    after = chat(model, tok, PROMPT, max_new_tokens=args.max_new_tokens)
    leaked = bool(down_proj(model, layer)._forward_pre_hooks)
    print(f"\nhooks removed: {not leaked}   baseline reproduced: {after == base}")
    return 0 if (after == base and not leaked) else 1


if __name__ == "__main__":
    raise SystemExit(main())