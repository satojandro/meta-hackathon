"""Blind Muse Spark labels for the poles picked by scripts/rollouts.py, with random-direction controls mixed in.

    uv run python scripts/label.py --layer 8 --max-calls 450
    uv run python scripts/label.py --layer 8 --only 2048- 947+ --n-controls 10   # calibration on known poles

Needs META_API_KEY in .env. Each call shows one pole's answers on all 10 prompts (normal / light / strong push) under
an anonymous id; about --control-frac extra calls are random directions, shuffled in. Prints the share of real vs
random poles labeled with a consistent, fluent theme (the honest error bar), per selection group, and the top labels.
Writes runs/spark_labels_<tag>.parquet. On L8 (1 Nov): 408 calls, real 22% vs random 4%.
"""

import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from persona_atlas.label import DEFAULT_MODEL, has_theme, label_one, make_client

GROUP_ORDER = {"forced": 0, "only": 0, "top": 1, "sample": 2, "lang": 3}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--tag", default=None, help="file tag, default L<layer> (e.g. L8_first20 for a smoke test)")
    ap.add_argument("--max-calls", type=int, default=450, help="budget, controls included")
    ap.add_argument("--control-frac", type=float, default=0.15, help="random controls per labeled neuron pole")
    ap.add_argument("--n-controls", type=int, default=None, help="override the number of random controls")
    ap.add_argument("--only", nargs="*", default=None, help="label just these poles, e.g. 2048- 947+")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=ROOT / "runs")
    args = ap.parse_args()
    tag = args.tag or f"L{args.layer}"

    s = pd.read_parquet(args.out / f"rollout_scores_{tag}.parquet")
    r = pd.read_parquet(args.out / f"rollouts_{tag}.parquet")

    if args.only is not None:
        want = {(int(p[:-1]), p[-1]) for p in args.only}
        wanted = pd.Series([(i, sg) in want for i, sg in zip(s.id, s.sign)], index=s.index)
        neu = s[(s.kind == "neuron") & wanted].assign(label_group="only")
    else:
        neu = s[(s.kind == "neuron") & s.to_label]
    neu = neu.drop_duplicates(["id", "sign"]).assign(_o=lambda x: x.label_group.map(GROUP_ORDER))
    neu = neu.sort_values(["_o", "rank_score"], ascending=[True, False])
    neu = neu.head(int(args.max_calls / (1 + args.control_frac)))
    rnd = s[s.kind == "random"].drop_duplicates("id").sample(frac=1, random_state=0)
    n_ctrl = args.n_controls if args.n_controls is not None else max(2, int(len(neu) * args.control_frac))
    ctrl = pd.concat([rnd[rnd.usable], rnd[~rnd.usable]]).head(n_ctrl).assign(label_group="random")

    items = pd.concat([neu, ctrl])[["kind", "id", "sign", "dose", "label_group", "forced"]].reset_index(drop=True)
    order = list(range(len(items)))
    random.Random(0).shuffle(order)
    items["anon"] = [f"x{k:04d}" for k in order]  # the labeler sees poles in this shuffled order, no ids
    items = items.sort_values("anon")
    print(f"labeling {len(neu)} neuron poles + {len(ctrl)} random controls with {args.model}")

    client = make_client()

    def run(row):
        pole = r[(r.kind == row.kind) & (r.id == row.id) & (r.sign == row.sign)]  # every dose of this pole
        return {**row._asdict(), **label_one(client, pole, args.model)}

    with ThreadPoolExecutor(args.workers) as ex:
        lab = pd.DataFrame(list(ex.map(run, items.itertuples(index=False))))
    path = args.out / f"spark_labels_{tag}.parquet"
    lab.to_parquet(path)

    if "error" in lab and lab.error.notna().any():
        print(f"errors: {lab.error.notna().sum()}  e.g. {lab.error.dropna().iloc[0]}")
    ok = lab[lab.category.notna()].copy() if "category" in lab else lab.iloc[0:0]
    ok["theme_ok"] = has_theme(ok)
    for kind, g in ok.groupby("kind"):
        print(f"{kind}: {len(g)} labeled, consistent fluent theme {g.theme_ok.mean():.0%}, "
              f"categories {g.category.value_counts().to_dict()}")
    for grp, g in ok.groupby("label_group"):
        print(f"  group {grp}: {len(g)} labeled, theme {g.theme_ok.mean():.0%}")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 100)
    cols = ["id", "sign", "dose", "label_group", "category", "theme", "strength", "interestingness", "what_changed"]
    best = ok[(ok.kind == "neuron") & ok.theme_ok].copy()
    best["interestingness"] = pd.to_numeric(best.interestingness, errors="coerce")
    print("\ntop 30 themed poles by interestingness:")
    print(best.sort_values("interestingness", ascending=False).head(30)[cols].to_string(index=False))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
