"""Rollouts for a screened layer: shortlist, generate real answers, drop junk, pick what the labeler sees.

    uv run python scripts/rollouts.py runs/kl_llama-3.2-1b-instruct_L8_broad.parquet --layer 8 --force 2048- 947+
    uv run python scripts/rollouts.py <screen.parquet> --layer 8 --limit 20 --n-random 10   # smoke test
    uv run python scripts/rollouts.py <screen.parquet> --layer 8 --rescore runs/rollouts_L8.parquet  # step 3 only

1. Shortlist: the top --top-pct of neuron poles within ANY prompt family, plus --force poles (known candidates, so
   the run doubles as a recall check).
2. Generate: every shortlisted pole at each --doses on the 8 screen prompts + 2 held-out ones, greedy, --max-new
   tokens, batched; plus --n-random size-matched random directions per dose and one unsteered baseline per prompt.
   Saves a checkpoint per prompt in runs/rollouts_<tag>.partial/, so a crash or restart resumes.
3. Score and pick: drop collapsed / unchanged / word-salad poles, keep language switches apart, then mark
   --n-top best-ranked + --n-sample random usable + --n-lang language poles + forced ones for labeling.

Writes runs/rollouts_<tag>.parquet (one row per pole x dose x prompt) and runs/rollout_scores_<tag>.parquet (one row
per pole x dose). On the 16 GB Mac Mini, L8 with ~2,900 poles took ~17 s per 128-row call, about 2 h 15 min.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch

from persona_atlas import load_model, residual_norms
from persona_atlas.model import chat_ids
from persona_atlas.rollouts import (
    HELD_OUT_PROMPTS, generate_batch, pick_for_label, score_rollouts, shortlist, unsteered_nll,
)
from persona_atlas.screen import PROMPT_SETS, neuron_vectors, random_vectors


def sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def free_cache(device) -> None:
    """Batched generate() left the MPS cache growing to 10 GB on the Mini, which pushed it into swap (18 s -> 150 s
    per call). Emptying it after every call keeps memory flat."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def parse_pole(s: str) -> tuple[int, str]:
    """'2048-' -> (2048, '-')."""
    return int(s[:-1]), s[-1]


def generate(args, short: pd.DataFrame, tag: str) -> pd.DataFrame:
    model, tok = load_model()
    resid = residual_norms(model, tok)[args.layer]
    prompts = PROMPT_SETS["broad"] + HELD_OUT_PROMPTS
    print(f"L{args.layer} resid {resid:.2f}; doses {args.doses}; {len(prompts)} prompts; {args.n_random} random/dose")

    units = [("neuron", int(i), s, d) for d in args.doses for i, s in zip(short.id, short.sign)]
    units += [("random", seed, "+", d) for d in args.doses for seed in range(args.n_random)]
    ckpt = args.out / f"rollouts_{tag}.partial"
    ckpt.mkdir(parents=True, exist_ok=True)
    todo = [(j, p, fam) for j, (p, fam) in enumerate(prompts) if not (ckpt / f"p{j}.parquet").exists()]
    n_calls = len(todo) * -(-len(units) // args.batch)
    print(f"{len(units)} pushes per prompt; {len(prompts) - len(todo)} of {len(prompts)} prompts already done")

    done, t0 = 0, time.perf_counter()
    for j, prompt, fam in todo:
        pids = chat_ids(tok, prompt).to(model.device)
        g, n = generate_batch(model, tok, pids, args.layer, None, args.max_new)
        baseline = tok.decode(g[0, : n[0]], skip_special_tokens=True).strip()
        rows = []
        for start in range(0, len(units), args.batch):
            chunk = units[start:start + args.batch]
            vecs = torch.stack([
                neuron_vectors(model, args.layer, [i], d if s == "+" else -d, resid)[0] if kind == "neuron"
                else random_vectors(model, args.layer, [i], d, resid)[0]
                for kind, i, s, d in chunk
            ])
            gen, lens = generate_batch(model, tok, pids, args.layer, vecs, args.max_new)
            nll = unsteered_nll(model, pids, gen, lens)
            for (kind, i, s, d), gi, li, ni in zip(chunk, gen, lens, nll):
                rows.append(dict(kind=kind, id=i, sign=s, dose=d, prompt=prompt, family=fam, nll=float(ni),
                                 text=tok.decode(gi[:li], skip_special_tokens=True).strip(), baseline=baseline))
            del gen, lens, vecs
            free_cache(model.device)
            done += 1
            if done == 1 or done % 10 == 0 or done == n_calls:
                sync(model.device)
                per = (time.perf_counter() - t0) / done
                print(f"  rollouts: call {done}/{n_calls}  {per:.1f}s/call  ~{per * (n_calls - done) / 60:.1f} min left",
                      flush=True)
        pd.DataFrame(rows).to_parquet(ckpt / f"p{j}.parquet")
    return pd.concat([pd.read_parquet(ckpt / f"p{j}.parquet") for j in range(len(prompts))], ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("screen", type=Path, help="screen parquet from scripts/screen.py")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--top-pct", type=float, default=10.0, help="keep the top N%% within any prompt family")
    ap.add_argument("--force", nargs="*", default=[], help="always include, e.g. 2048- 947+")
    ap.add_argument("--doses", type=float, nargs="+", default=[0.5, 0.8])
    ap.add_argument("--n-random", type=int, default=100, help="random-direction controls per dose")
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None, help="only the first N shortlisted poles (smoke test)")
    ap.add_argument("--rescore", type=Path, default=None, help="skip generation: rescore an existing rollouts parquet")
    ap.add_argument("--n-top", type=int, default=100, help="best-ranked usable poles to label")
    ap.add_argument("--n-sample", type=int, default=300, help="random sample of the other usable poles to label")
    ap.add_argument("--n-lang", type=int, default=30, help="language switchers to label")
    ap.add_argument("--max-novelty", type=float, default=4.0, help="median unsteered NLL above this = word salad")
    ap.add_argument("--out", type=Path, default=ROOT / "runs")
    args = ap.parse_args()

    forced = [parse_pole(p) for p in args.force]
    short = shortlist(pd.read_parquet(args.screen), args.top_pct, forced)
    if args.limit:
        short = pd.concat([short[short.forced], short[~short.forced].head(args.limit)])
    print(f"shortlist: {len(short)} poles (top {args.top_pct}% in any prompt family; forced {forced})")
    tag = f"L{args.layer}" + (f"_first{args.limit}" if args.limit else "")

    if args.rescore:
        r = pd.read_parquet(args.rescore)
    else:
        r = generate(args, short, tag)
        path = args.out / f"rollouts_{tag}.parquet"
        r.to_parquet(path)
        print(f"wrote {path}  ({len(r)} rollouts)")

    s = score_rollouts(r, max_novelty=args.max_novelty)
    s = s.merge(short[["id", "sign", "forced", "best_family_pct", "top_family", "median_kl"]], on=["id", "sign"],
                how="left")
    s.loc[s.kind == "random", ["forced", "best_family_pct", "top_family", "median_kl"]] = [False, None, None, None]
    s["forced"] = s.forced.fillna(False).astype(bool)
    s = pick_for_label(s, args.n_top, args.n_sample, args.n_lang, forced)
    s["to_label"] = s.label_group.notna()
    path = args.out / f"rollout_scores_{tag}.parquet"
    s.to_parquet(path)

    neu, rnd = s[s.kind == "neuron"], s[s.kind == "random"]
    for d in sorted(s.dose.unique()):
        x, y = neu[neu.dose == d], rnd[rnd.dose == d]
        print(f"dose {d}: neurons usable {x.usable.mean():.0%} (collapsed {(x.collapsed > 0.5).mean():.0%}, "
              f"salad {(x.novelty > args.max_novelty).mean():.0%}, lang {x.lang.mean():.0%}) | random usable "
              f"{y.usable.mean():.0%} | median novelty neurons {x.novelty.median():.2f} vs random {y.novelty.median():.2f}")
    cols = ["id", "sign", "dose", "collapsed", "changed", "novelty", "rank_score", "label_group"]
    print("\nforced poles:")
    print(neu[neu.forced].sort_values(["id", "dose"])[cols].to_string(index=False))
    print(f"\nto label: {s.label_group.value_counts().to_dict()}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
