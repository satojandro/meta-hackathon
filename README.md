# meta-hackathon

Entry for the **Meta Global AI Developer Hackathon** (Cerebral Valley, 26 Oct – 4 Nov 2026), Indie developer track.

**Project (in progress): a persona-neuron atlas for Muse Glimmer.** A Muse Spark agent maps the neurons inside
Meta's open-weights Muse Glimmer 30B that change the model's sense of who it is, and a persona mixer lets you build
characters by turning neuron dials instead of writing prompts. See [NEURON_PROJECT.md](NEURON_PROJECT.md).

## Setup

Runs on an Apple Silicon Mac (PyTorch MPS) or any CUDA GPU, with the same commands — so the laptop and a rented
GPU behave identically.

**1. uv** (installs Python for you; no separate Python install needed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # macOS/Linux
uv --version                                       # verified with 0.11.19
```

`.python-version` pins 3.11 and uv fetches that interpreter on first sync.

**2. Code and environment**

```bash
git clone https://github.com/satojandro/meta-hackathon && cd meta-hackathon
uv sync      # creates .venv: torch 2.15 + transformers 5.22, ~840 MB, about a minute
```

**3. Model and keys** (`.env`)

```bash
cp .env.example .env
```

| Variable | Needed for | Notes |
|---|---|---|
| `MODEL_ID` | which weights to load | default in `.env.example` is `unsloth/Llama-3.2-1B-Instruct`, an ungated mirror of `meta-llama/Llama-3.2-1B-Instruct` (same 1.24B weights, same licence). No token needed. |
| `HF_TOKEN` | gated Meta repos | only if you switch `MODEL_ID` to the official `meta-llama/...` repo: read-scope token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) plus licence approval. |
| `META_API_KEY` | Muse Spark (hosted research agent / labeller) | from [dev.meta.ai](https://dev.meta.ai). Not needed to run local steering. |
| `META_API_BASE` | — | `https://api.meta.ai/v1` (OpenAI-compatible). |
| `FIREWORKS_API_KEY` | optional | hosted Glimmer / Spark Open for bulk labelling. |

Local steering needs **no keys at all** — a two-line `.env` is enough:

```ini
MODEL_ID=unsloth/Llama-3.2-1B-Instruct
META_API_BASE=https://api.meta.ai/v1
```

**4. Verify**

```bash
uv run pytest                    # 6 checks on a tiny random Llama: ~10 s, no download
uv run python scripts/smoke.py   # real model: downloads 2.5 GB once, then answers "Who are you?",
                                 # steers three neurons at ±0.8 with random controls, and re-checks the baseline
```

**5. What fits where**

| Machine | 1B dev model (fp16, 2.5 GB) | Glimmer 30B |
|---|---|---|
| MacBook (M4, 24 GB) | yes, this is the dev loop | no |
| Mac Mini (M4, 16 GB) | yes | no |
| Modal A100/H100 | yes | this is where the sweep runs |

Budget ~3.3 GB of disk per machine (venv + weights) and one 2.5 GB download.

**6. Known quirks**

- `transformers` 5.x returns a `BatchEncoding` from `apply_chat_template`, not a bare tensor; `model.chat_ids`
  handles both (fixed 30 Oct).
- bf16 on Apple MPS is unreliable, so `default_dtype` uses fp16 on MPS and bf16 on CUDA.
- `HF_HUB_OFFLINE=1` makes a run fail loudly rather than silently download.
- If a Hugging Face download stalls part-way (seen on this network: 2.2 GB of 2.5 GB, then 0 bytes/s, and
  `hf_transfer` restarted from zero), resume the prefix instead of re-downloading: copy the `.incomplete` blob to
  `models/llama-3.2-1b-instruct/model.safetensors`, then
  `curl -L -C - --retry 30 --retry-all-errors --speed-limit 50000 --speed-time 20 -o <same path> <resolve url>`.
  Check `shasum -a 256` against the `lfs.sha256` in `https://huggingface.co/api/models/<repo>?blobs=true`, copy the
  tokenizer/config JSONs out of the cache snapshot, and point `MODEL_ID` at the directory.

Code lives in `src/persona_atlas/`:

- `model.py`: load a model on CUDA / MPS / CPU, greedy chat replies.
- `calibrate.py`: median residual-stream norm per layer (the unit for steering doses).
- `steering.py`: push one MLP neuron (or a size-matched random control) at the input to `down_proj`.
- `scripts/smoke.py`: end-to-end check on whatever machine you are on.

## Built during the hackathon
All code and content in this repository was written during the event (from 30 Oct 2026). Commit history is the record.

## Credits and prior art
Ideas and methodology (no code) drawn from:
- [lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer): steering-first neuron explorer for Qwen3.5-4B
- Persona-neuron sweep of Qwen3.5-4B shared on X
- [neuronary.org](https://neuronary.org): community dictionary of neuron steering recipes

Details in [PRIOR_ART.md](PRIOR_ART.md).
