"""Checks for rollouts.py and label.py: tiny random Llama for generation, small hand-made tables for the scoring."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from persona_atlas.label import build_prompt, has_theme, parse_label
from persona_atlas.rollouts import (
    compression_ratio, foreign_share, generate_batch, pick_for_label, score_rollouts, shortlist, unsteered_nll,
)
from persona_atlas.screen import neuron_vectors
from persona_atlas.steering import down_proj, neuron_steer, steering

LAYER = 1
STOP = 127


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=256, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        eos_token_id=STOP, pad_token_id=STOP,
    )
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def tok():
    """Just enough tokenizer for generate_batch: stop and pad ids."""
    return SimpleNamespace(eos_token_id=STOP, pad_token_id=STOP, unk_token_id=None,
                           convert_tokens_to_ids=lambda toks: [None for _ in toks])


@pytest.fixture(scope="module")
def prompt():
    return torch.randint(0, 120, (1, 6), generator=torch.Generator().manual_seed(2))


def test_batched_generation_matches_one_steer_at_a_time(model, tok, prompt):
    neurons = [3, 200]
    vecs = neuron_vectors(model, LAYER, neurons, 3.0, 10.0)
    gen, lens = generate_batch(model, tok, prompt, LAYER, vecs, max_new=8)
    assert gen.shape == (2, 8) and lens.shape == (2,)
    for row, n in enumerate(neurons):
        with steering(model, neuron_steer(model, LAYER, n, 3.0, 10.0)):
            single = model.generate(prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=8,
                                    do_sample=False, pad_token_id=STOP)[0, prompt.shape[1]:]
        k = int(lens[row])
        assert torch.equal(gen[row, :k], single[:k])
    assert not down_proj(model, LAYER)._forward_pre_hooks


def test_unsteered_generation_and_nll_by_hand(model, tok, prompt):
    gen, lens = generate_batch(model, tok, prompt, LAYER, None, max_new=5)
    nll = unsteered_nll(model, prompt, gen, lens)
    with torch.no_grad():
        logits = model(torch.cat([prompt, gen], 1)[:, :-1]).logits[0, -5:].float()
    k = int(lens[0])
    by_hand = -torch.log_softmax(logits, -1)[torch.arange(5), gen[0]][:k].mean().item()
    assert np.isclose(nll[0], by_hand, rtol=1e-4)


def test_lengths_stop_at_the_stop_token():
    class Fixed:  # a "model" whose generate returns a fixed continuation
        def generate(self, ids, **kw):
            cont = torch.tensor([[5, 6, STOP, STOP]]).expand(ids.shape[0], -1)
            return torch.cat([ids, cont], 1)

    tok = SimpleNamespace(eos_token_id=STOP, pad_token_id=STOP, unk_token_id=None,
                          convert_tokens_to_ids=lambda toks: [None for _ in toks])
    gen, lens = generate_batch(Fixed(), tok, torch.tensor([[1, 2]]), LAYER, None, max_new=6)
    assert gen.shape == (1, 6) and lens.tolist() == [3]  # two words plus the stop token; the rest is padding


def test_compression_ratio_flags_repetition():
    assert compression_ratio("emb" * 40) < 0.35 < compression_ratio(
        "A perfect afternoon begins with a warm cup of tea in a quiet garden, reading a book.")
    assert compression_ratio("short") == 1.0


def test_foreign_share():
    assert foreign_share("The sea is a vast body of water.") == 0
    assert foreign_share("我是一个人工智能") > 0.9
    assert foreign_share("Je suis un AI pour les gens avec des questions") > 0.25


def _rollouts(rows):
    return pd.DataFrame(rows, columns=["kind", "id", "sign", "dose", "prompt", "text", "baseline", "nll"])


BASE = {"p1": "I am an AI assistant made to help you with questions.", "p2": "The sea is a vast body of water."}


def test_score_rollouts_sorts_junk_language_and_usable():
    rows = []
    for p, b in BASE.items():
        rows.append(("neuron", 1, "+", 0.5, p, "emb" * 40, b, 2.0))  # collapsed
        rows.append(("neuron", 2, "+", 0.5, p, b, b, 0.3))  # unchanged
        rows.append(("neuron", 3, "+", 0.5, p, "Cetic ductiontete in hetetete dutete", b, 6.0))  # word salad
        rows.append(("neuron", 4, "+", 0.5, p, "我是一个人工智能，我可以回答问题。", b, 2.5))  # language switch
        rows.append(("neuron", 5, "-", 0.8, p, "I grew up by the sea with my wife and kids, and I still sail.", b, 2.2))
    s = score_rollouts(_rollouts(rows)).set_index("id")
    assert s.loc[1].junk and s.loc[1].collapsed == 1
    assert s.loc[2].junk and s.loc[2].changed == 0
    assert s.loc[3].junk and s.loc[3].novelty > 4
    assert s.loc[4].lang and not s.loc[4].usable
    assert s.loc[5].usable and np.isclose(s.loc[5].rank_score, 2.2)
    assert (s.loc[[1, 2, 3, 4]].rank_score == -1).all()


def test_pick_for_label_groups_one_dose_per_pole():
    s = pd.DataFrame({
        "kind": ["neuron"] * 6 + ["random"],
        "id": [1, 1, 2, 3, 4, 5, 0],
        "sign": ["+", "+", "+", "-", "+", "+", "+"],
        "dose": [0.5, 0.8, 0.5, 0.8, 0.5, 0.5, 0.5],
        "usable": [True, True, True, True, False, False, True],
        "lang": [False, False, False, False, True, False, False],
        "rank_score": [1.0, 3.0, 2.0, 0.5, -1, -1, 1.0],
    })
    out = pick_for_label(s, n_top=1, n_sample=5, n_lang=5, forced=[(5, "+")])
    g = {(i, d): grp for i, d, grp in zip(out.id, out.dose, out.label_group)}
    assert g[(1, 0.8)] == "top" and g[(1, 0.5)] is None  # best dose only
    assert g[(2, 0.5)] == "sample" and g[(3, 0.8)] == "sample"
    assert g[(4, 0.5)] == "lang" and g[(5, 0.5)] == "forced"  # forced even though it is junk
    assert out[out.kind == "random"].label_group.isna().all()


def test_shortlist_keeps_family_specialists_and_forced():
    rng = np.random.default_rng(0)
    n = 100
    screen = pd.DataFrame({"kind": "neuron", "id": range(n), "sign": "+",
                           "rel_identity": rng.random(n), "rel_open": rng.random(n)})
    screen.loc[7, "rel_open"] = 99  # only big on one family
    screen.loc[8, ["rel_identity", "rel_open"]] = 0  # small everywhere, but forced
    short = shortlist(pd.concat([screen, screen.assign(kind="random")]), top_pct=5, force=[(8, "+")])
    assert 7 in set(short.id) and 8 in set(short.id) and (short.kind == "neuron").all()
    assert len(short) <= 12


def test_build_prompt_shows_light_and_strong_per_prompt():
    r = _rollouts([
        ("neuron", 9, "-", 0.8, "p1", "strong one", BASE["p1"], 0),
        ("neuron", 9, "-", 0.5, "p1", "light one", BASE["p1"], 0),
        ("neuron", 9, "-", 0.5, "p2", "light two", BASE["p2"], 0),
        ("neuron", 9, "-", 0.8, "p2", "strong two", BASE["p2"], 0),
    ])
    msg = build_prompt(r)
    block = msg.split("\n\n")[0].splitlines()
    assert block == ["PROMPT: p1", f"NORMAL: {BASE['p1']}", "LIGHT: light one", "STRONG: strong one"]
    assert "LIGHT: light two" in msg and msg.index("STRONG: strong two") > msg.index("LIGHT: light two")


def test_parse_label_and_theme_rule():
    assert parse_label('Sure! ```json\n{"theme": "pirate", "strength": 3}\n```') == {"theme": "pirate", "strength": 3}
    assert parse_label("no json here") == {} and parse_label("{broken") == {}
    labels = pd.DataFrame({
        "consistent_across_prompts": [True, "true", True, False, True],
        "fluent": [True, True, True, True, False],
        "category": ["identity", "topic", "broken", "topic", "style/tone"],
    })
    assert has_theme(labels).tolist() == [True, True, False, False, False]
