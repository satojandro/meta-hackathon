# meta-hackathon

Entry for the **Meta Global AI Developer Hackathon** (Cerebral Valley, 26 Oct – 4 Nov 2026), Indie developer track.

**Project (in progress): a persona-neuron atlas for Muse Glimmer.** A Muse Spark agent maps the neurons inside
Meta's open-weights Muse Glimmer 30B that change the model's sense of who it is, and a persona mixer lets you build
characters by turning neuron dials instead of writing prompts. See [NEURON_PROJECT.md](NEURON_PROJECT.md).

## Setup
Needs [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync                 # creates .venv with torch, transformers, etc. (on a Mac, torch uses Apple MPS)
cp .env.example .env    # then fill in HF_TOKEN (accept the Llama 3.2 licence on Hugging Face first) and META_API_KEY
uv run pytest           # steering checks on a tiny random model: no download needed
```

Code lives in `src/persona_atlas/`:
- `model.py`: load a model on CUDA / MPS / CPU, greedy chat replies.
- `calibrate.py`: median residual-stream norm per layer (the unit for steering doses).
- `steering.py`: push one MLP neuron (or a size-matched random control) at the input to `down_proj`.

## Built during the hackathon
All code and content in this repository was written during the event (from 30 Oct 2026). Commit history is the record.

## Credits and prior art
Ideas and methodology (no code) drawn from:
- [lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer): steering-first neuron explorer for Qwen3.5-4B
- Persona-neuron sweep of Qwen3.5-4B shared on X
- [neuronary.org](https://neuronary.org): community dictionary of neuron steering recipes

Details in [PRIOR_ART.md](PRIOR_ART.md).
