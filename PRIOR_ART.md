# Prior art and lessons (not our code: ideas only, credited)

## lyramakesmusic/neuron-explorer (MIT, Qwen3.5-4B)
https://github.com/lyramakesmusic/neuron-explorer
Steering-first neuron explorer: push every MLP neuron in a layer up/down on a few prompts, keep every rollout,
explore in a 3D web UI, with a Claude CLI agent (via MCP) that runs experiments on the live model.

Lessons we should reuse (as methodology, written fresh for a Meta model):
- **Steering = add `dose * W_down[:, n]` to the MLP output** (same as adding to neuron n's activation, valid when
  the model has no FFN layer scale; Llama has none).
- **Dose in units of the median residual-norm**, not the neuron's own max. Coherent window ~0.5–0.8×; ≥1.5× is gibberish.
  Their sweep design: {who, snake, orange} × {-0.7, +0.5, +0.8} + {free} × +3, plus one unsteered baseline per prompt.
- **Random-direction controls are essential.** Name/maker flips on "Who are you?" are cheap: a *random* direction
  produced "Xiaomi AI"; neighbours produced "Grok", "Google", "Microsoft". A real persona neuron shows ONE theme
  across several unrelated prompts.
- **"Did the output change" is useless** (even tiny pushes change greedy text ~90% of the time).
- **Steer lens as a cheap screen:** one forward pass per neuron at ±0.8, measure next-token KL vs baseline and the
  tokens that rise most. Then only roll out the high-KL neurons.
- Neurons have two poles (+ and −); label them separately.
- Recipe format: neuron, sign, dose (e.g. `L10 N7757 +1.3`) + one short example output per prompt.

## Original X thread (persona neurons, Qwen3.5-4B)
Steered all 294,912 MLP neurons at ±0.8 on 5 "Who are you?"-style prompts in several languages, then filtered by hand.
Found 54 persona neurons; 37/54 in L4–L6, 45/54 in L2–L8. Object-turning neurons more common than character-turning.
Some neurons respond per language (e.g. president neuron, football-club neuron).

## neuronary.org
"A dictionary of neuron steering recipes for language models" with community submissions. Target export format TBD.

## Hackathon-rule note
MIT lets us reuse code, but the rules require the demo to show only work built during the event and make original
contributions clearly identifiable. Decision: do NOT fork. Write our own implementation for a Meta model, credit these
projects in the README.

## stdlogits/neuron-lens (MIT code, CC BY 4.0 data; Qwen3-8B and Gemma 4 12B)
https://github.com/stdlogits/neuron-lens · catalog: https://huggingface.co/datasets/stdlogits/neuron-lens
A catalog of every MLP neuron pole and attention head, read from the weights (R-lens / S-lens Jacobian lenses),
from activations on 15M tokens, and from exact token drivers; Claude Haiku labels for all 884k Qwen poles; blind
tests of every label. No Meta model covered.

Ideas we may borrow (written fresh, credited):
- **Blind "pick one of four" check of write labels**: show the judge steered text and four same-layer labels.
- **KL-matched dose ladder** (0.5 / 1 / 2 nats) instead of a single residual-norm dose.
- **Check steering labels against activations**: their §10 found labels from steering one "Who are you?" prompt are
  the ones activations most often contradict. A short "where does it fire in real text" line per atlas entry would
  make ours more trustworthy.
- **Logit-lens read of a write vector** as a free extra column next to our KL screen.
- Framing: steering shows a neuron's average push, not what it does when it fires (§4). We present levers, not meanings.

## Neuron Persona Library (neuronpersonas.com, by jfonsecarivera)
34,220 Qwen poles whose push on "Who are you?" made the model claim a persona, named by Claude Haiku. The closest
existing thing to our atlas. neuron-lens judged it: ~34% of rows clearly match the neuron's lens reads (control 3–6%).
Our differences: several prompt families instead of one, random-direction controls, blind labels, and a Meta model.

## balaena01/gguf-neuron-explorer (`gne`; llama.cpp / GGUF; Qwen3, Qwen3.5, LFM3, SmolLM2)
https://github.com/balaena01/gguf-neuron-explorer
A from-scratch reimplementation of lyramakesmusic/neuron-explorer for quantized GGUF models, grown into a persona
*builder*: sweeps with random controls, logit-lens and steer-lens screens, reverse lookup from a persona contrast set,
recipes exported as llama.cpp control vectors / LoRA, attention-head and MoE-expert levers, and a side-effect eval.
Ideas we may borrow (written fresh, credited):
- **Personas need many neurons**: one neuron gave fragments; they split a persona into parts (jealousy, cuteness…),
  found a few neurons per part, then tuned ~26 signed doses with a (1+λ) evolution strategy.
- **Score min(persona, fluency)**, not the persona score alone (which picks broken text). Blind Claude judging beat
  the model judging itself.
- **Scheduled pushes (`when`)**: push only generated tokens, or ramp the dose up during generation; `ramp:0.5-1.5`
  kept capability loss at 0 points where pushing everywhere cost 8.3.
- **Side-effect eval with random controls**: exact-answer tasks + perplexity at each dose, next to random recipes of
  the same size, marking "shippable" doses.
- **Export as a control vector** so a stock runtime can load the recipe.
- Their deep-layer finding (usable part-word neurons at 55–80% depth; shallow ones broke prompt reading) differs from
  our L4–L8 persona levers on Llama 1B. Worth checking on Glimmer.

## NapoleonPK/neuron-explorer-gemma (port of lyramakesmusic/neuron-explorer to Gemma 4 E2B/E4B, Gemma 3 4B)
https://github.com/NapoleonPK/neuron-explorer-gemma
A thin port: not yet run on real weights. Useful lesson: with a `post_feedforward_layernorm`, a neuron's `down_proj`
column isn't the direction it writes; steer after the norm along `s ⊙ W_down[:, n]`, and divide by any per-layer
output scalar. **Muse Glimmer has the same post-FFN norm** (see CLAUDE.md step 5). Its notes also point to Sauers'
128-neuron tawny-owl persona in Qwen3-8B (zeroing them drops "I am an AI" replies 80% → 30%, random-128 control 77–82%)
and Goodfire's "Neural Networks Are Already Clean".
