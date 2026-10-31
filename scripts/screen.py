"""KL steer-lens screen over one layer: which neurons, pushed at +-dose, change what the model is about to say?

    uv run python scripts/screen.py --layer 4 --limit 256      # quick timing run on the first 256 neurons
    uv run python scripts/screen.py --layer 4                  # the full layer
    uv run python scripts/screen.py --layer 8 --show 2048      # also print one neuron's row
    uv run python scripts/screen.py --layer 4 --prompts identity --dose 1.0 --n-answer 12   # the 31 Oct setup

For each prompt (default: the "broad" set, 2 prompts each from 4 families): one batched forward pass per batch of
neurons, KL(steered || baseline) over the first --n-answer tokens of the model's own answer, for every neuron at
+dose and -dose. The same for size-matched random directions gives a null. A pole is a hit if it beats the null's
99th percentile on ~60% of the prompts (3 of 5, 5 of 8).

Hits are then sorted twice: by raw median KL, and as "coherent hits" whose push does not collapse the model onto one
repeated token (collapse < --max-collapse), each with the prompt family it moves most.

Writes runs/kl_<model>_L<layer>[_<prompt set>][_d<dose>].parquet: one row per neuron pole (kind="neuron") or random
control (kind="random"), with per-prompt KL, max KL, top-token mass and top risers, plus median_kl, collapse,
neuron_pct, per-family KL relative to the null (rel_<family>), top_family, prompts_beaten, hit and coherent.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from persona_atlas import load_model, residual_norms
from persona_atlas.screen import (
    PROMPT_SETS, default_min_prompts, kl_batch, make_baseline, neuron_percentile, neuron_vectors, random_vectors,
    score_poles,
)
from persona_atlas.steering import down_proj, get_layers


# The 31 Oct dose sweep on Llama 3.2 1B: concept neurons read as fluent text around 0.35-0.5 and collapse into one
# repeated token by 0.65-1.0, so the screen pushes at 0.5 rather than the 1.0 it started with.
DEFAULT_DOSE = 0.5


def sync(device) -> None:
    """Wait for queued GPU work, so timings measure the work and not just the queueing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--dose", type=float, default=DEFAULT_DOSE, help="screened at +dose and -dose")
    ap.add_argument("--prompts", choices=sorted(PROMPT_SETS), default="broad", help="which prompt set to screen on")
    ap.add_argument("--min-prompts", type=int, default=None, help="hit rule: prompts to beat (default ~60%%)")
    ap.add_argument("--max-collapse", type=float, default=0.5,
                    help="a coherent hit puts less than this probability on one repeated token")
    ap.add_argument("--n-random", type=int, default=256, help="random controls for the null")
    ap.add_argument("--batch", type=int, default=128, help="pushes per forward pass; lower if out of memory")
    ap.add_argument("--n-answer", type=int, default=24,
                    help="teacher-forced answer tokens to score; more catches changes that start later in the answer")
    ap.add_argument("--limit", type=int, default=None, help="only the first N neurons (timing / quick runs)")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--show", type=int, nargs="*", default=[], help="neuron ids to print whatever their score")
    ap.add_argument("--out", type=Path, default=ROOT / "runs")
    args = ap.parse_args()

    model, tok = load_model()
    layer = args.layer
    d_mlp = down_proj(model, layer).weight.shape[1]
    neurons = list(range(d_mlp if args.limit is None else min(args.limit, d_mlp)))
    model_name = (getattr(model.config, "name_or_path", "") or model.config.model_type).split("/")[-1].lower()
    print(f"model {model_name}  device {model.device}  layer {layer}/{len(get_layers(model))}  d_mlp {d_mlp}")

    resid = residual_norms(model, tok)[layer]
    print(f"L{layer} median residual norm {resid:.2f}; dose +-{args.dose} = MLP output shift {args.dose * resid:.2f}")

    prompt_set = PROMPT_SETS[args.prompts]
    prompts = [p for p, _ in prompt_set]
    families = [f for _, f in prompt_set]
    min_prompts = args.min_prompts or default_min_prompts(len(prompts))
    bases = [make_baseline(model, tok, p, args.n_answer) for p in prompts]
    for b, fam in zip(bases, families):
        print(f"  [{fam:8}] {b.prompt!r:44} -> {tok.decode(b.answer_ids)!r}")

    def sweep(vectors: torch.Tensor, label: str):
        """Every row of `vectors` on every prompt.

        Returns mean KL, max KL, top-token mass [rows, prompts] and risers [rows, prompts, k].
        """
        n, p = vectors.shape[0], len(bases)
        mean_kl, max_kl, top_mass = np.zeros((n, p)), np.zeros((n, p)), np.zeros((n, p))
        risers = np.zeros((n, p, 5), dtype=np.int64)
        n_batches = -(-n // args.batch) * p
        done, t0 = 0, time.perf_counter()
        for j, base in enumerate(bases):
            for start in range(0, n, args.batch):
                res = kl_batch(model, base, layer, vectors[start:start + args.batch])
                rows = slice(start, start + len(res.mean_kl))
                mean_kl[rows, j], max_kl[rows, j], risers[rows, j] = res.mean_kl, res.max_kl, res.risers
                top_mass[rows, j] = res.top_mass
                done += 1
                if done == 1 or done % 20 == 0 or done == n_batches:
                    sync(model.device)
                    per = (time.perf_counter() - t0) / done
                    print(f"  {label}: batch {done}/{n_batches}  {per:.2f}s/batch  ~{per * (n_batches - done) / 60:.1f} min left", flush=True)
        return mean_kl, max_kl, top_mass, risers

    print(f"\nrandom null: {args.n_random} size-matched directions")
    null = sweep(random_vectors(model, layer, range(args.n_random), args.dose, resid), "null")

    print(f"\nneurons: {len(neurons)} x 2 signs")
    vecs = torch.cat([neuron_vectors(model, layer, neurons, s * args.dose, resid) for s in (1, -1)])
    poles = sweep(vecs, "neurons")
    assert not down_proj(model, layer)._forward_pre_hooks, "a steering hook leaked"

    thresholds, beaten, hit = score_poles(poles[0], null[0], min_prompts=min_prompts)
    _, null_beaten, null_hit = score_poles(null[0], null[0], min_prompts=min_prompts)
    fam_names = list(dict.fromkeys(families))

    def frame(kind, ids, signs, stats, beaten, hit):
        mean_kl, max_kl, top_mass, risers = stats
        df = pd.DataFrame({"kind": kind, "id": ids, "sign": signs})
        for j in range(len(bases)):
            df[f"kl_p{j}"] = mean_kl[:, j]
            df[f"maxkl_p{j}"] = max_kl[:, j]
            df[f"topmass_p{j}"] = top_mass[:, j]
            df[f"risers_p{j}"] = [" ".join(repr(tok.decode([t])) for t in r) for r in risers[:, j]]
        df["median_kl"] = np.median(mean_kl, axis=1)
        df["collapse"] = np.median(top_mass, axis=1)
        # KL in units of the null's p99, averaged per family: > 1 means the family moved more than chance allows.
        rel = mean_kl / thresholds
        for fam in fam_names:
            df[f"rel_{fam}"] = rel[:, [f == fam for f in families]].mean(axis=1)
        df["top_family"] = df[[f"rel_{f}" for f in fam_names]].to_numpy().argmax(axis=1)
        df["top_family"] = [fam_names[i] for i in df["top_family"]]
        df["prompts_beaten"], df["hit"] = beaten, hit
        df["coherent"] = hit & (df["collapse"] < args.max_collapse)
        # Risers from the prompt this pole moved most: the best single hint at its theme.
        best = mean_kl.argmax(axis=1)
        df["risers"] = [df.at[i, f"risers_p{b}"] for i, b in enumerate(best)]
        return df

    neuron_frame = frame("neuron", neurons * 2, ["+"] * len(neurons) + ["-"] * len(neurons), poles, beaten, hit)
    neuron_frame["neuron_pct"] = neuron_percentile(poles[0])
    df = pd.concat([
        neuron_frame,
        frame("random", list(range(args.n_random)), "+", null, null_beaten, null_hit),
    ], ignore_index=True)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"kl_{model_name}_L{layer}"
    stem += "" if args.prompts == "identity" else f"_{args.prompts}"
    stem += "" if args.dose == DEFAULT_DOSE else f"_d{args.dose:g}"
    stem += "" if args.limit is None else f"_first{len(neurons)}"
    path = args.out / f"{stem}.parquet"
    df.to_parquet(path)
    meta = {
        "model": model_name, "layer": layer, "dose": args.dose, "resid_norm": resid, "n_answer": args.n_answer,
        "n_neurons": len(neurons), "n_random": args.n_random, "prompt_set": args.prompts, "prompts": prompts,
        "families": families, "min_prompts": min_prompts, "max_collapse": args.max_collapse,
        "answers": [tok.decode(b.answer_ids) for b in bases], "null_p99": thresholds.tolist(),
        "hits": int(hit.sum()), "hit_fraction": float(hit.mean()),
        "random_hits": int(null_hit.sum()), "coherent_hits": int(neuron_frame["coherent"].sum()),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    print(f"\nnull p99 per prompt: {' '.join(f'{t:.3f}' for t in thresholds)}")
    print(f"hits: {hit.sum()} of {len(hit)} poles ({100 * hit.mean():.1f}%), beating the null on >= {min_prompts} "
          f"of {len(prompts)} prompts.  Random directions passing the same rule: {null_hit.sum()} of {len(null_hit)}.")
    neu = df[df.kind == "neuron"]
    coherent = neu[neu.coherent]
    print(f"coherent hits (collapse < {args.max_collapse}): {len(coherent)}.  "
          f"Hits by top family: {neu[neu.hit].top_family.value_counts().to_dict()}")
    cols = ["id", "sign", "median_kl", "collapse", "top_family", "prompts_beaten", "risers"]
    print(f"\ntop {args.top} hits by median KL:")
    print(neu[neu.hit].sort_values("median_kl", ascending=False).head(args.top)[cols].to_string(index=False))
    print(f"\ntop {args.top} coherent hits by median KL:")
    print(coherent.sort_values("median_kl", ascending=False).head(args.top)[cols].to_string(index=False))
    if args.show:
        print("\nrequested neurons:")
        print(neu[neu.id.isin(args.show)][cols + ["neuron_pct", "hit", "coherent"]].to_string(index=False))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
