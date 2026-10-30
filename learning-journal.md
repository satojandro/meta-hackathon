# Learning journal

Things I've learned during the Meta hackathon. Newest entries at the bottom of each day.
Format: **Concept**: plain-English explanation. *Why it matters* / source.

---

## Fri 30 Oct 2026: getting oriented

### How a language model is built inside
- **Latent space / hidden state**: as each word (token) passes through the model, it is represented as a long list of
  numbers (a vector), e.g. 3,072 or 6,656 numbers. That vector is the model's "thought" about the token at that point.
  "Latent" = hidden; we never see it in the output.
- **Residual stream**: the main highway carrying that vector through the model. Each layer *reads* from it and *adds*
  something back. Interpretability is mostly about figuring out what gets added and why.
- **Layers**: a model is a stack of identical blocks (Glimmer has 52). Each block = an **attention** part (mixes
  information between words) + an **MLP** part (processes each word on its own, where much "knowledge" lives).
- **MLP neuron**: one unit inside an MLP layer. Modern models use **SwiGLU** MLPs: each neuron computes
  `silu(gate · x) × (up · x)` and then writes its value along its own fixed output direction (`W_down[:, n]`)
  into the residual stream. Qwen3.5-4B has 294,912 of them; Glimmer has far more.
- **Two poles**: a neuron can be pushed positive or negative, and the two directions often mean different things.
  Treat `L10 N7757+` and `L10 N7757−` as separate objects.

### Neuron steering
- **Steering** = adding a little extra to one neuron while the model generates, then reading what it writes.
  Mechanically: add `dose × W_down[:, n]` to that layer's MLP output.
- **Dose units matter**: measure the push relative to the typical size (norm) of the residual stream. Around
  0.5–0.8× keeps text coherent; above ~1.5× the model produces gibberish.
- **Persona neurons**: some single neurons flip the model's identity when pushed ("Who are you?" → "I am Extrali, an
  extraterrestrial…"). In Qwen3.5-4B, 37 of 54 found persona neurons sat in early layers L4–L6.
- **Polysemanticity**: many neurons respond to several unrelated things, which makes them messy to label. That's why
  researchers also use SAEs (sparse autoencoders) to find cleaner "features".

### Doing it like a scientist
- **Random-direction controls** (the null hypothesis): push a *random* direction of the same size and see what
  happens. In Qwen, a random direction made the model say it was "Xiaomi AI". So a name change alone proves nothing.
  A real persona neuron shows the *same theme across unrelated prompts*.
- **"Did the output change?" is a useless test**: even tiny pushes change greedy output ~90% of the time.
- **Steer lens / KL screen**: a cheap first pass. Push each neuron, run *one* forward pass, and measure how much the
  next-token probabilities shift (KL divergence). Only generate full answers for neurons that move things a lot.
  This turns a weeks-long brute force into hours.

### Models and hardware
- **Hosted vs open weights**: Muse Spark (API) is closed: you can only send text in and get text out, so you can't
  look inside. Muse Glimmer and Muse Spark Open are open weights: you download them and can hook into every neuron.
- **Dense vs Mixture of Experts (MoE)**: in a dense model (Glimmer, 30B) every parameter runs on every token.
  An MoE (Spark Open: 816B total, 42B active, 256 experts, 8 used per token) routes each token to a few
  "expert" sub-networks. MoE neurons only matter when their expert is chosen, which makes neuron mapping much harder.
- **Memory math**: BF16 = 2 bytes per parameter → 30B params ≈ 60 GB. 4-bit ≈ 0.5 bytes → ≈ 18 GB.
  Spark Open in BF16 ≈ 1.6 TB, so it needs a multi-GPU server.
- **Distillation**: training a small model to imitate a big one. Glimmer was distilled from Muse Spark, so mapping
  Glimmer is a realistic way to peek at the Spark family.

### Choosing which model to map
- **Why not map Muse Spark (the hosted API model)?** It's closed. Through the API you only get text in and text out:
  no weights, no activations, no way to hook a neuron. Interpretability needs the weights on a machine you control.
- **Why not map Muse Spark Open (the open-weights version)?** Possible in principle, but not in six days:
  - **Size:** 816B parameters. Even the 4-bit (NVFP4) version is ~484 GB, so it needs 4–8 top-end GPUs
    (~$32/hr on Modal for 8× H100; the $100 credit buys ~3 hours, including load time).
  - **Architecture:** it's a Mixture of Experts (256 experts, 8 used per token). Each expert's neurons only fire when
    the router picks that expert, so "push every neuron and watch" becomes tens of millions of *conditional* neurons.
    You'd need a new method (e.g. steering whole experts or the router) before you could even start.
  - **No existing tooling:** the steering code everyone uses assumes dense models.
- **Why Muse Glimmer 30B is the better target:**
  - **Dense**: every neuron runs on every token, so the proven brute-force steering method works as-is.
  - **Same MLP style (SwiGLU)** as Qwen, so lessons from the Qwen work transfer directly.
  - **Fits one GPU**: ~60 GB in BF16 → a single A100/H100 80GB on Modal (~$2.50–3.95/hr).
  - **Apache 2.0 license**: no restrictions on publishing the map.
  - **Distilled from Muse Spark**: mapping Glimmer is the realistic way to peek inside the Spark family. Story:
    "the Spark agent studies the neurons of the model it taught."
  - **New and unmapped**: released Aug 2026; no public neuron map exists yet, so any finding is new.
  - **Gotcha to remember**: Glimmer uses Gemma-style norms *after* the MLP, so steer at the neuron itself (the input
    to `down_proj`) rather than adding to the MLP output.

### Hackathon craft
- Judges must be able to tell what you built *during* the event, so start from an empty repo and credit prior work
  (ideas, not code).
- An MIT license allows reuse, but the hackathon rules are stricter than the license.

### Sources
- lyramakesmusic/neuron-explorer: https://github.com/lyramakesmusic/neuron-explorer (research_log.md is gold)
- neuronary.org: a community dictionary of neuron steering recipes
- ARENA (Alignment Research Engineer Accelerator): mechanistic interpretability curriculum

### Compute inside the weights: Qwenputer
- **What it is**: Qwen3.5-4B with a full 32-bit RISC-V CPU *compiled* into extra layers in front of the model.
  It runs C and MicroPython inside ordinary forward passes, with no external tool. Arithmetic accuracy matches
  "Qwen + a real Python tool" (e.g. NUPA long-integer maths 48.6% → 100%).
  Source: https://divagr.com/blog/qwenputer · code: https://github.com/divagr18/qwenputer (MIT)
- **Compiled, not trained**: no gradient descent. A compiler turns each CPU step into weights directly: attention
  heads do lookups, MLP units do arithmetic. Transformers are general enough to *be* a program, not just learn one.
- **Gated DeltaNet (GDN)**: a "linear attention" layer type. Instead of a growing memory of every past token
  (a KV cache), it keeps a fixed-size **state matrix** updated each token by the *delta rule*. Qwen3.5 mixes GDN layers
  with normal attention.
- **GDN as RAM**: set the forget gate to 1 (nothing fades) and the write strength to 1 (full overwrite), and use
  one-hot keys. Each write then replaces exactly one row and each read returns exactly one row: a perfect memory cell.
  Registers, program counter and memory all live in those rows.
- **Piggybacking on tool calls**: Qwen already knows how to write `<tool_call>` blocks. The compiled layers
  intercept the call, run it on the CPU, and the model writes the `<tool_response>` itself, digit by digit.
- **Execution trace as tokens**: each CPU instruction becomes a few generated tokens (`ADDR`, `VAL`, `STEP`).
  The model's own decoding loop *is* the CPU clock, which is why it's slow (~3,200 trace tokens/s on an L4).
- **"Did we break the model?" check**: they measured KL divergence between normal text from base vs modified Qwen
  (6e-4 vs a 5e-4 noise floor), meaning the model's normal behaviour is essentially unchanged. Same KL tool as the
  neuron steer lens, used for the opposite purpose.

### Why Qwenputer doesn't port to Meta's models (this week)
- **No DeltaNet in Glimmer**: Glimmer uses standard (grouped-query + sliding-window) attention. Without GDN's
  overwritable fixed-size state, there's no ready-made RAM, so memory would have to be rebuilt from attention over
  the trace tokens. That's a redesign, not a port.
- **Scale of the original**: the repo is ~200k lines and a long research effort. Re-implementing it in six days for a
  new architecture isn't realistic, and reusing its code would clash with the hackathon's "new work only" rule.
- **Takeaway**: architecture details (which layer types a model has) decide which interpretability or compilation
  tricks are even possible. Always check the architecture first.
