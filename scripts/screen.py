"""KL steer-lens screen over one layer: which neurons, pushed at +-dose, change what the model is about to say?

    uv run python scripts/screen.py --layer 4 --limit 256      # quick timing run on the first 256 neurons
    uv run python scripts/screen.py --layer 4                  # the full layer
    uv run python scripts/screen.py --layer 8 --show 2048      # also print one neuron's row

For each of 5 identity prompts: one batched forward pass per batch of neurons, KL(steered || baseline) over the first
~12 tokens of the model's own answer, for every neuron at +dose and -dose. The same for size-matched random
directions gives a null. A pole is a hit if it beats the null's 99th percentile on 3 of 5 prompts.

Writes runs/kl_<model>_L<layer>.parquet: one row per neuron pole (kind="neuron") or random control
(kind="random"), with per-prompt KL, max KL and top risers, plus median_kl, prompts_beaten and hit.
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
    IDENTITY_PROMPTS, kl_batch, make_baseline, neuron_vectors, random_vectors, score_poles,
)
from persona_atlas.steering import down_proj, get_layers


def sync(device) -> None:
    """Wait for queued GPU work, so timings measure the work and not just the queueing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--dose", type=float, default=1.0, help="screened at +dose and -dose")
    ap.add_argument("--n-random", type=int, default=256, help="random controls for the null")
    ap.add_argument("--batch", type=int, default=128, help="pushes per forward pass; lower if out of memory")
    ap.add_argument("--n-answer", type=int, default=12, help="teacher-forced answer tokens to score")
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

    bases = [make_baseline(model, tok, p, args.n_answer) for p in IDENTITY_PROMPTS]
    for b in bases:
        print(f"  {b.prompt!r:40} -> {tok.decode(b.answer_ids)!r}")

    def sweep(vectors: torch.Tensor, label: str):
        """Every row of `vectors` on every prompt. Returns mean KL, max KL [rows, prompts], risers [rows, prompts, k]."""
        n, p = vectors.shape[0], len(bases)
        mean_kl, max_kl, risers = np.zeros((n, p)), np.zeros((n, p)), np.zeros((n, p, 5), dtype=np.int64)
        n_batches = -(-n // args.batch) * p
        done, t0 = 0, time.perf_counter()
        for j, base in enumerate(bases):
            for start in range(0, n, args.batch):
                res = kl_batch(model, base, layer, vectors[start:start + args.batch])
                rows = slice(start, start + len(res.mean_kl))
                mean_kl[rows, j], max_kl[rows, j], risers[rows, j] = res.mean_kl, res.max_kl, res.risers
                done += 1
                if done == 1 or done % 20 == 0 or done == n_batches:
                    sync(model.device)
                    per = (time.perf_counter() - t0) / done
                    print(f"  {label}: batch {done}/{n_batches}  {per:.2f}s/batch  ~{per * (n_batches - done) / 60:.1f} min left", flush=True)
        return mean_kl, max_kl, risers

    print(f"\nrandom null: {args.n_random} size-matched directions")
    null = sweep(random_vectors(model, layer, range(args.n_random), args.dose, resid), "null")

    print(f"\nneurons: {len(neurons)} x 2 signs")
    vecs = torch.cat([neuron_vectors(model, layer, neurons, s * args.dose, resid) for s in (1, -1)])
    poles = sweep(vecs, "neurons")
    assert not down_proj(model, layer)._forward_pre_hooks, "a steering hook leaked"

    thresholds, beaten, hit = score_poles(poles[0], null[0])
    _, null_beaten, null_hit = score_poles(null[0], null[0])

    def frame(kind, ids, signs, stats, beaten, hit):
        mean_kl, max_kl, risers = stats
        df = pd.DataFrame({"kind": kind, "id": ids, "sign": signs})
        for j in range(len(bases)):
            df[f"kl_p{j}"] = mean_kl[:, j]
            df[f"maxkl_p{j}"] = max_kl[:, j]
            df[f"risers_p{j}"] = [" ".join(repr(tok.decode([t])) for t in r) for r in risers[:, j]]
        df["median_kl"] = np.median(mean_kl, axis=1)
        df["prompts_beaten"], df["hit"] = beaten, hit
        # Risers from the prompt this pole moved most: the best single hint at its theme.
        best = mean_kl.argmax(axis=1)
        df["risers"] = [df.at[i, f"risers_p{b}"] for i, b in enumerate(best)]
        return df

    df = pd.concat([
        frame("neuron", neurons * 2, ["+"] * len(neurons) + ["-"] * len(neurons), poles, beaten, hit),
        frame("random", list(range(args.n_random)), "+", null, null_beaten, null_hit),
    ], ignore_index=True)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"kl_{model_name}_L{layer}{'' if args.limit is None else f'_first{len(neurons)}'}.parquet"
    df.to_parquet(path)
    meta = {
        "model": model_name, "layer": layer, "dose": args.dose, "resid_norm": resid, "n_answer": args.n_answer,
        "n_neurons": len(neurons), "n_random": args.n_random, "prompts": IDENTITY_PROMPTS,
        "answers": [tok.decode(b.answer_ids) for b in bases], "null_p99": thresholds.tolist(),
        "hits": int(hit.sum()), "hit_fraction": float(hit.mean()),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    print(f"\nnull p99 per prompt: {' '.join(f'{t:.3f}' for t in thresholds)}")
    print(f"hits: {hit.sum()} of {len(hit)} poles ({100 * hit.mean():.1f}%).  "
          f"Random directions pass the same 3-of-5 rule well under 1% of the time, so a rate near that is chance.")
    neu = df[df.kind == "neuron"]
    cols = ["id", "sign", "median_kl", "prompts_beaten", "risers"]
    print(f"\ntop {args.top} hits by median KL:")
    print(neu[neu.hit].sort_values("median_kl", ascending=False).head(args.top)[cols].to_string(index=False))
    if args.show:
        print("\nrequested neurons:")
        print(neu[neu.id.isin(args.show)][cols + ["hit"]].to_string(index=False))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
