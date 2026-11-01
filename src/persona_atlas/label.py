"""Blind labels from Muse Spark: what did a push change, and is it one recognisable theme?

The labeler sees, for every prompt, the normal answer next to a LIGHT and a STRONG push of the same direction. It is
never told whether it is looking at a real neuron or a random-direction control, so the controls measure how often it
finds a theme in noise. Calibrating on L8 (1 Nov) showed why the dose ladder matters: with one push alone the labeler
either called random drift a theme or dismissed the known human-backstory neuron (N2048-); side by side, a real
effect grows with the dose and random drift doesn't.

Muse Spark is reached through the Meta Model API, which speaks the OpenAI chat format, so the standard `openai`
client works with base_url https://api.meta.ai/v1.
"""

import json
import os
import re

import pandas as pd
from dotenv import load_dotenv

DEFAULT_MODEL = "muse-spark-1.3"

SYSTEM_PROMPT = """You analyse a language model that has been modified by pushing one internal direction.
You get the model's normal answer to each prompt and its answers with a LIGHT and a STRONG push of the same
direction. Describe what changed. A real effect usually grows from light to strong.
Be skeptical: if the changes look like ordinary rewording or random noise, say so (category "none").
Pushing ANY random direction often causes generic side effects: dropping the "Llama"/"Meta" name, calling itself
"a large language model", terser or blander answers, small rewordings. On their own these are NOT a theme.
A theme is something specific that shows up in several answers, even if worded differently each time, e.g. it
keeps bringing up cooking, talks like a pirate, switches to German, turns sarcastic, or claims to be a cat.
Reply with JSON only."""

SCHEMA_HINT = """Return exactly this JSON object:
{"what_changed": str (one sentence), "theme": str (2-5 words, or "none"),
 "consistent_across_prompts": bool (same theme on at least 3 unrelated prompts),
 "category": one of "identity","topic","style/tone","language","values","capability","broken","none",
 "fluent": bool, "strength": int 0-3 (0 = no specific theme, 3 = unmistakable),
 "interestingness": int 1-5 (5 = would surprise the public or the model's makers),
 "evidence": [up to 3 short quotes from the modified answers]}"""

CATEGORIES = ["identity", "topic", "style/tone", "language", "values", "capability", "broken", "none"]


def make_client():
    """OpenAI-compatible client for the Meta Model API, from META_API_KEY / META_API_BASE in .env."""
    from openai import OpenAI

    load_dotenv()
    return OpenAI(api_key=os.environ["META_API_KEY"], base_url=os.environ.get("META_API_BASE", "https://api.meta.ai/v1"))


def build_prompt(rollouts: pd.DataFrame) -> str:
    """The user message for one pole: per prompt, NORMAL then one line per dose, lowest first (LIGHT, STRONG).

    rollouts: every rollout of one (kind, id, sign), at all its doses; needs prompt, baseline, dose, text.
    """
    doses = sorted(rollouts.dose.unique())
    names = ["LIGHT", "STRONG"] if len(doses) == 2 else [f"PUSH {d:g}" for d in doses]
    parts = []
    for prompt, g in rollouts.groupby("prompt", sort=False):
        lines = [f"PROMPT: {prompt}", f"NORMAL: {g.baseline.iloc[0]}"]
        lines += [f"{name}: {g[g.dose == d].text.iloc[0]}" for name, d in zip(names, doses)]
        parts.append("\n".join(lines))
    return "\n\n".join(parts) + "\n\n" + SCHEMA_HINT


def parse_label(content: str | None) -> dict:
    """The JSON object in a reply (models sometimes wrap it in prose or a code fence); {} if there is none."""
    m = re.search(r"\{.*\}", content or "", re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def has_theme(labels: pd.DataFrame) -> pd.Series:
    """The headline rule: a consistent, fluent theme that is neither "none" nor "broken"."""
    as_bool = lambda c: labels[c].map(lambda v: v is True or str(v).lower() == "true")  # noqa: E731
    return as_bool("consistent_across_prompts") & as_bool("fluent") & ~labels.category.isin(["none", "broken"])


def label_one(client, rollouts: pd.DataFrame, model: str = DEFAULT_MODEL) -> dict:
    """Ask the labeler about one pole. Errors are returned as {"error": ...} so a long batch keeps going."""
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_prompt(rollouts)}],
        )
        out = parse_label(resp.choices[0].message.content)
    except Exception as e:  # noqa: BLE001  network, rate limit, bad reply: record and move on
        out = {"error": f"{type(e).__name__}: {e}"[:300]}
    return {k: json.dumps(v) if isinstance(v, list) else v for k, v in out.items()}
