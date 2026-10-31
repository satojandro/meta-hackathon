"""Checks for the batched KL screen, on the same tiny random Llama as test_steering.py."""

import copy

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from persona_atlas.screen import (
    PROMPT_SETS, Baseline, batched_steering, default_min_prompts, kl_batch, neuron_percentile, neuron_vectors,
    random_vectors, score_poles,
)
from persona_atlas.steering import down_proj, neuron_steer, random_steer, steering

LAYER = 1
N_ANSWER = 4


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=256, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def base(model):
    """A fake teacher-forced sequence: random 'prompt + answer' ids and the unsteered log-probs along them."""
    ids = torch.randint(0, 128, (1, 10), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        logits = model(ids, logits_to_keep=N_ANSWER).logits[0]
    return Baseline("test", ids, ids[0, -N_ANSWER:], torch.log_softmax(logits.float(), dim=-1))


def test_batched_row_matches_single_neuron_steer(model, base):
    neurons = [3, 7, 200]
    vecs = neuron_vectors(model, LAYER, neurons, 0.8, 10.0)
    with torch.no_grad(), batched_steering(model, LAYER, vecs):
        batched = model(base.input_ids.expand(len(neurons), -1)).logits
    for i, n in enumerate(neurons):
        with torch.no_grad(), steering(model, neuron_steer(model, LAYER, n, 0.8, 10.0)):
            single = model(base.input_ids).logits[0]
        assert torch.allclose(batched[i], single, atol=1e-5)
    assert not down_proj(model, LAYER)._forward_pre_hooks


def test_random_vectors_match_random_steer(model):
    vecs = random_vectors(model, LAYER, [0, 5], 0.8, 10.0)
    assert torch.equal(vecs[1], random_steer(model, LAYER, 0.8, 10.0, seed=5).vector)


def test_zero_push_has_zero_kl(model, base):
    res = kl_batch(model, base, LAYER, torch.zeros(2, 256))
    assert np.allclose(res.mean_kl, 0, atol=1e-6) and np.allclose(res.max_kl, 0, atol=1e-6)


def test_kl_matches_a_by_hand_computation_and_grows_with_dose(model, base):
    vecs = torch.cat([neuron_vectors(model, LAYER, [7], d, 10.0) for d in (0.5, 2.0)])
    res = kl_batch(model, base, LAYER, vecs, k=3, chunk=1)
    with torch.no_grad(), steering(model, neuron_steer(model, LAYER, 7, 2.0, 10.0)):
        logp = torch.log_softmax(model(base.input_ids, logits_to_keep=N_ANSWER).logits[0].float(), dim=-1)
    kl = (logp.exp() * (logp - base.logprobs)).sum(-1)
    assert np.isclose(res.mean_kl[1], kl.mean().item(), rtol=1e-4)
    assert np.isclose(res.max_kl[1], kl.max().item(), rtol=1e-4)
    assert res.mean_kl[1] > res.mean_kl[0] > 0
    assert res.risers.shape == (2, 3)
    assert np.isclose(res.top_mass[1], logp.exp().mean(0).max().item(), rtol=1e-4)


def test_top_mass_flags_collapse_onto_one_token(model, base):
    """A push that drives the model to the same token at every position gives top_mass near 1.

    A random tiny model's logits are too flat to collapse, so wire one in: neuron 7 writes token 5's unembedding
    direction, and a sharper lm_head turns that into a confident prediction.
    """
    m = copy.deepcopy(model)
    with torch.no_grad():
        m.lm_head.weight.mul_(50)
        down_proj(m, LAYER).weight[:, 7] = m.lm_head.weight[5]
        logits = m(base.input_ids, logits_to_keep=N_ANSWER).logits[0]
    b = Baseline("test", base.input_ids, base.answer_ids, torch.log_softmax(logits.float(), dim=-1))
    vecs = torch.cat([torch.zeros(1, 256), neuron_vectors(m, LAYER, [7], 50.0, 10.0)])
    res = kl_batch(m, b, LAYER, vecs)
    assert res.top_mass[0] < 0.5 < res.top_mass[1] <= 1.0


def test_neuron_percentile_ranks_by_median_kl():
    kl = np.array([[1.0, 1, 1], [5, 5, 5], [3, 3, 3], [0, 9, 9]])  # medians 1, 5, 3, 9
    pct = neuron_percentile(kl)
    assert pct.argsort().tolist() == [0, 2, 1, 3]
    assert 0 < pct.min() < pct.max() < 100


def test_prompt_sets_and_default_bar():
    assert default_min_prompts(5) == 3 and default_min_prompts(8) == 5
    assert len(PROMPT_SETS["identity"]) == 5
    fams = {f for _, f in PROMPT_SETS["broad"]}
    assert fams == {"identity", "open", "task", "opinion"} and len(PROMPT_SETS["broad"]) == 8


def test_hit_rule_needs_three_of_five_prompts():
    null = np.tile(np.arange(100.0)[:, None], (1, 5))  # p99 = 98.01 on every prompt
    poles = np.array([
        [200, 200, 200, 0, 0],  # beats the null on 3 prompts -> hit
        [200, 200, 0, 0, 0],  # only 2 -> not a hit
        [98, 98, 98, 98, 98],  # just under the threshold -> not a hit
    ], dtype=float)
    thresholds, beaten, hit = score_poles(poles, null)
    assert np.allclose(thresholds, 98.01)
    assert beaten.tolist() == [3, 2, 0]
    assert hit.tolist() == [True, False, False]
