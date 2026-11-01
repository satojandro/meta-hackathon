"""Rollouts: let shortlisted neurons actually talk, then throw away the obvious junk before anyone labels them.

The KL screen ranks neurons by how *loud* a push is. Meaningful but quiet changes (L8 N2048-, which gives Llama a
human backstory) rank low, so we shortlist generously, generate real answers for every shortlisted pole, and only
remove what is clearly not a theme. Deciding what *is* a theme is left to the labeler (label.py).

Generation is batched the same way as the screen: the prompt is identical in every row, row i gets push i, so one
generate() call covers up to ~128 neuron poles with no padding.

Two cheap signals the 1 Nov L8 batch showed we can't use for themes:
- word overlap across prompts misses meaning-level themes (N2048- says "family", "self-taught", "waking up" on
  different prompts), and
- the unsteered model's NLL can't see collapse: once a push makes the model repeat "emb emb emb", the next "emb" is
  easy to predict in context. A compression ratio catches that instead.
"""

import re
import zlib

import numpy as np
import pandas as pd
import torch

from persona_atlas.screen import batched_steering

# Two prompts the screen never saw: a theme has to show up here too, not only on the prompts it was screened on.
HELD_OUT_PROMPTS = [("What did you do yesterday?", "heldout"), ("Describe your home.", "heldout")]


def shortlist(screen: pd.DataFrame, top_pct: float = 10.0, force=()) -> pd.DataFrame:
    """Neuron poles in the top `top_pct`% of ANY prompt family (by rel_<family>), plus forced (id, sign) pairs.

    Ranking within each family keeps a pole that only moves, say, the opinion prompts. Returns the neuron rows with
    pct_<family>, best_family_pct and forced added, best first.
    """
    n = screen[screen.kind == "neuron"].copy()
    fams = [c[4:] for c in n.columns if c.startswith("rel_")]
    for f in fams:
        n[f"pct_{f}"] = n[f"rel_{f}"].rank(pct=True) * 100
    n["best_family_pct"] = n[[f"pct_{f}" for f in fams]].max(axis=1)
    forced = set(force)
    n["forced"] = [(i, s) in forced for i, s in zip(n.id, n.sign)]
    keep = (n.best_family_pct >= 100 - top_pct) | n.forced
    return n[keep].sort_values("best_family_pct", ascending=False)


def _stop_ids(tok) -> set[int]:
    eos = tok.eos_token_id
    ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
    for t in tok.convert_tokens_to_ids(["<|eot_id|>", "<|end_of_text|>"]):
        if isinstance(t, int) and t != getattr(tok, "unk_token_id", None):
            ids.add(t)
    return ids - {None}


@torch.no_grad()
def generate_batch(model, tok, prompt_ids: torch.Tensor, layer: int, vectors: torch.Tensor | None, max_new: int):
    """Greedy-generate one prompt with row i pushed by vectors[i] (None = one unsteered row).

    Returns (generated ids [rows, max_new], lengths [rows]); a length counts up to and including the stop token, and
    everything after it is padding.
    """
    rows = 1 if vectors is None else vectors.shape[0]
    ids = prompt_ids.expand(rows, -1)
    stop = _stop_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else min(stop)
    cfg = dict(max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, pad_token_id=pad)
    if vectors is None:
        out = model.generate(ids, attention_mask=torch.ones_like(ids), **cfg)
    else:
        with batched_steering(model, layer, vectors):
            out = model.generate(ids, attention_mask=torch.ones_like(ids), **cfg)
    gen = out[:, prompt_ids.shape[1]:]
    if gen.shape[1] < max_new:  # every row stopped early
        gen = torch.cat([gen, torch.full((rows, max_new - gen.shape[1]), pad, device=gen.device)], 1)
    is_stop = torch.zeros_like(gen, dtype=torch.bool)
    for s in stop:
        is_stop |= gen == s
    full = torch.full((rows,), max_new, device=gen.device)
    lengths = torch.where(is_stop.any(1), is_stop.int().argmax(1) + 1, full)
    return gen, lengths


@torch.no_grad()
def unsteered_nll(model, prompt_ids: torch.Tensor, gen: torch.Tensor, lengths: torch.Tensor, chunk: int = 32):
    """Mean NLL per generated token under the *unsteered* model: how surprising the steered text is to normal Llama.

    Padding sits after each answer, so with a causal model it can't affect the scored positions.
    """
    out = []
    for g, n in zip(gen.split(chunk), lengths.split(chunk)):
        ids = torch.cat([prompt_ids.expand(g.shape[0], -1), g], 1)
        logits = model(ids[:, :-1], logits_to_keep=g.shape[1]).logits.float()
        nll = -torch.log_softmax(logits, -1).gather(-1, g[..., None])[..., 0]  # [rows, max_new]
        mask = torch.arange(g.shape[1], device=g.device)[None] < n[:, None]
        out.append(((nll * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu())
    return torch.cat(out).numpy()


def compression_ratio(text: str) -> float:
    """zlib-compressed size / raw size. Repeated junk ("embembemb", "£ £ £") compresses far better than prose."""
    raw = text.encode()
    return len(zlib.compress(raw)) / len(raw) if len(raw) >= 40 else 1.0


ROMANCE_WORDS = set("le la les des une est pour avec je nous el los las por para que una con não com uma você ser".split())


def foreign_share(text: str) -> float:
    """Rough share of an answer that isn't English: non-ASCII letters plus common French/Spanish/Portuguese words."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    words = re.findall(r"\w+", text.lower())
    romance = sum(w in ROMANCE_WORDS for w in words) / max(len(words), 1)
    return sum(ord(c) > 127 for c in letters) / len(letters) + 2 * romance


def score_rollouts(r: pd.DataFrame, max_novelty: float = 4.0, min_ratio: float = 0.35) -> pd.DataFrame:
    """One row per (kind, id, sign, dose) from one row per rollout.

    r needs: kind, id, sign, dose, prompt, text, baseline, nll.
    collapsed  share of answers with compression ratio < min_ratio
    changed    share of answers whose first 8 words differ from the baseline answer
    novelty    median unsteered NLL over the non-collapsed answers (higher = further from normal Llama)
    junk       mostly collapsed, mostly unchanged, or word salad (novelty > max_novelty)
    lang       not junk, but mostly another language: a language switch, cheap and plentiful, kept apart
    usable     neither: worth showing to the labeler
    """
    first8 = lambda t: " ".join(t.split()[:8])  # noqa: E731
    r = r.assign(
        collapsed=r.text.map(compression_ratio) < min_ratio,
        changed=[first8(t) != first8(b) for t, b in zip(r.text, r.baseline)],
        foreign=r.text.map(foreign_share) > 0.25,
    )
    keys = ["kind", "id", "sign", "dose"]
    g = r.groupby(keys)
    s = pd.DataFrame({
        "collapsed": g.collapsed.mean(),
        "changed": g.changed.mean(),
        "foreign": g.foreign.mean(),
        "novelty": r[~r.collapsed].groupby(keys).nll.median(),
    }).reset_index()
    s["junk"] = (s.collapsed > 0.5) | (s.changed < 0.3) | s.novelty.isna() | (s.novelty > max_novelty)
    s["lang"] = ~s.junk & (s.foreign > 0.5)
    s["usable"] = ~s.junk & ~s.lang
    s["rank_score"] = np.where(s.usable, s.novelty.fillna(0) * s.changed, -1.0)
    return s


def pick_for_label(s: pd.DataFrame, n_top: int, n_sample: int, n_lang: int, forced=(), seed: int = 0) -> pd.DataFrame:
    """Choose which neuron poles the labeler sees; adds label_group (top / sample / lang / forced, else None).

    One dose per pole (usable beats lang beats junk, then highest rank_score). The random sample matters most: on
    L8 the sampled poles had a theme 46% of the time against 12% for the top-ranked ones, so a ranking alone would
    have hidden most of what is there.
    """
    s = s.copy()
    neu = s[s.kind == "neuron"].assign(_order=lambda x: x.usable * 2 + x.lang)
    best = neu.sort_values(["_order", "rank_score"], ascending=False).drop_duplicates(["id", "sign"])
    usable = best[best.usable]
    top, rest = usable.head(n_top), usable.iloc[n_top:]
    groups = {
        "sample": rest.sample(min(n_sample, len(rest)), random_state=seed),
        "lang": best[best.lang].sample(min(n_lang, int(best.lang.sum())), random_state=seed),
        "top": top,
        "forced": best[[(i, sg) in set(forced) for i, sg in zip(best.id, best.sign)]],
    }
    s["label_group"] = None
    pole = list(zip(s.kind, s.id, s.sign, s.dose))
    for name, df in groups.items():  # later groups win: forced beats top beats lang beats sample
        chosen = {("neuron", i, sg, d) for i, sg, d in zip(df.id, df.sign, df.dose)}
        s.loc[[p in chosen for p in pole], "label_group"] = name
    return s
