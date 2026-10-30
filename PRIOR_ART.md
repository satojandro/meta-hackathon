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
