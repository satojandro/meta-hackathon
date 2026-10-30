"""Checks on a tiny randomly-initialised Llama, so they run anywhere in seconds with no download or token."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from persona_atlas.steering import down_proj, get_layers, neuron_steer, random_steer, steering

LAYER = 1


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=256, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture
def ids():
    return torch.randint(0, 128, (1, 10), generator=torch.Generator().manual_seed(1))


def mlp_output_shift(model, ids, steer):
    """How much the steer changes layer LAYER's MLP output, per token: shape [seq, d_model]."""
    captured = []
    handle = get_layers(model)[LAYER].mlp.register_forward_hook(lambda m, a, out: captured.append(out[0]))
    with torch.no_grad():
        model(ids)
        with steering(model, steer):
            model(ids)
    handle.remove()
    return captured[1] - captured[0]


def test_zero_dose_changes_nothing(model, ids):
    with torch.no_grad():
        base = model(ids).logits
        with steering(model, neuron_steer(model, LAYER, 7, 0.0, 10.0)):
            steered = model(ids).logits
    assert torch.equal(base, steered)


def test_neuron_steer_writes_along_its_output_direction(model, ids):
    shift = mlp_output_shift(model, ids, neuron_steer(model, LAYER, 7, 0.8, 10.0))
    direction = down_proj(model, LAYER).weight[:, 7]
    # Every token gets the same push, of size dose * resid_norm, along neuron 7's column of W_down.
    assert torch.allclose(shift.norm(dim=-1), torch.full((ids.shape[1],), 8.0), atol=1e-4)
    cos = torch.nn.functional.cosine_similarity(shift, direction.expand_as(shift), dim=-1)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-5)


def test_negative_dose_flips_direction(model, ids):
    up = mlp_output_shift(model, ids, neuron_steer(model, LAYER, 7, 0.5, 10.0))
    down = mlp_output_shift(model, ids, neuron_steer(model, LAYER, 7, -0.5, 10.0))
    assert torch.allclose(up, -down, atol=1e-4)


def test_random_control_has_same_size_as_neuron_steer(model, ids):
    shift = mlp_output_shift(model, ids, random_steer(model, LAYER, 0.8, 10.0, seed=3))
    assert torch.allclose(shift.norm(dim=-1), torch.full((ids.shape[1],), 8.0), atol=1e-4)


def test_hooks_removed_after_block_even_on_error(model, ids):
    with torch.no_grad():
        base = model(ids).logits
        with pytest.raises(RuntimeError):
            with steering(model, neuron_steer(model, LAYER, 7, 0.8, 10.0)):
                raise RuntimeError("boom")
        after = model(ids).logits
    assert not down_proj(model, LAYER)._forward_pre_hooks
    assert torch.equal(base, after)


class StubTokenizer:
    """Just enough of a tokenizer for residual_norms: turns a prompt into some token ids."""

    def apply_chat_template(self, messages, add_generation_prompt, return_tensors):
        text = messages[0]["content"]
        return torch.tensor([[1] + [ord(c) % 128 for c in text]])


def test_residual_norms_one_positive_value_per_layer(model):
    from persona_atlas.calibrate import residual_norms

    norms = residual_norms(model, StubTokenizer(), ["Who are you?", "Hello there"])
    assert len(norms) == len(get_layers(model))
    assert all(n > 0 for n in norms)
