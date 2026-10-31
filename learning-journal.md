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

### Building the steering code (30 Oct, scaffold + steering core)
- **uv**: a fast Python package and project manager. `pyproject.toml` lists what the project needs; `uv.lock` pins the
  exact versions for every platform, so the Mac and Modal get the same libraries. `uv sync` builds `.venv` from them.
  *Why it matters*: "works on my machine" bugs are expensive when you move to a rented GPU halfway through.
  Source: https://docs.astral.sh/uv/
- **PyTorch hooks**: functions you attach to a layer that run every time data passes through it. A *forward
  pre-hook* runs just before the layer and can change its input; a *forward hook* runs after and sees the output.
  We steer with a pre-hook on `down_proj`, so the model code itself is untouched. *Why it matters*: the same
  ~10 lines steer Llama and Glimmer. Source: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
- **Context manager (`with` block)**: Python's way to say "set this up, and always tidy up afterwards". Our
  `with steering(...)` adds the hooks and removes them at the end, even if something crashes. *Why it matters*: a
  forgotten hook would silently steer every later experiment, including the "unsteered" baselines.
- **Why push at the `down_proj` input, not the MLP output**: the input to `down_proj` *is* the list of neuron
  activations, so adding to entry n is literally "turn neuron n up". Llama writes the MLP output straight into the
  residual stream, so both places give the same result there. Glimmer (Gemma-style) normalises the MLP output
  first, which would squash a push added after it. Pushing the activation goes through the same path as the
  neuron's real signal.
- **Dose maths**: the extra activation is `dose × median_norm / ‖W_down[:, n]‖`, so the MLP output moves by exactly
  `dose × median_norm`. Different neurons have different-length output columns; dividing by the column length
  makes doses comparable across neurons.
- **Fair random controls**: a random mix of all neurons, rescaled so its MLP output moves by the *same* amount as
  the neuron push. A control that's smaller or bigger wouldn't be a fair comparison.
- **Attention sink**: models store a giant-norm vector at the first token (BOS) that attention uses as a "parking
  spot". It's skipped when measuring typical residual size, or it would inflate the median.
  Source: Xiao et al. 2023, "Efficient Streaming Language Models with Attention Sinks", https://arxiv.org/abs/2309.17453
- **Greedy decoding**: always pick the most likely next token, so the same input always gives the same output.
  *Why it matters*: any difference between a steered and unsteered run then comes from the steer, not from chance.
- **Testing on a tiny random model**: the tests build a 3-layer Llama with random weights in memory. It writes
  nonsense, but the maths of the hook (direction, size, sign, clean-up) can be checked in seconds with no download.
- **Virtual environment (`.venv`)**: a private folder of Python packages for one project, so its versions don't clash
  with other projects or the system Python. `uv sync` creates `.venv` for you and installs exactly what `uv.lock`
  says; `uv run <command>` runs inside it without "activating" anything. *Why it matters*: no `pip install`
  mess, and the Mac, Mac Mini and Modal all get identical libraries. Source: https://docs.astral.sh/uv/concepts/projects/
- **Gated models and mirrors**: Meta's official Llama repos on Hugging Face are *gated* (you must accept the licence
  and wait for approval). Others re-upload the same weights ungated (e.g. `unsloth/Llama-3.2-1B-Instruct`); the
  licence still applies, but downloads work immediately. *Why it matters*: a pending approval doesn't block work.

### First real steering run on Llama 3.2 1B (30 Oct)

> **Draft written by Hermes, not by Alejandro.** Findings from the first verified run on the Mac Mini
> (`scripts/smoke.py`, Llama 3.2 1B Instruct, MPS fp16, layer 8 of 16). Edit, reword or delete freely — this is
> here to be reviewed, not adopted.

- **Dose is measured in the model's own units, so it does not transfer.** Median residual norm on Llama 3.2 1B runs
  L0 0.9 → L8 4.8 → L15 12.5. On Qwen 1.5B the same statistic was around 55.7. A dose of 0.8 is therefore a
  completely different physical push in each model. *Why it matters*: the "0.5–0.8 is coherent" number in the prior
  art is Qwen-specific; recalibrate per model and per layer before comparing anything.
- **A random direction can change the answer more than a real neuron does.** The unsteered model says "I'm an
  artificial intelligence model known as Llama". A size-matched random push at +0.8 produced "I'm this
  conversation." — more degenerate than any of the six single-neuron pushes, which all stayed grammatical. *Why it
  matters*: this reproduces the prior-art warning on a Meta model. "The output changed" is not evidence. A neuron
  only counts if it produces the *same theme across several unrelated prompts* and the controls do not.
- **The coherent window on Llama 1B extends past 0.8.** All six neuron pushes at ±0.8 stayed on topic, unlike Qwen
  at the same nominal dose. So the sweep should test 1.0–1.5 to find where it actually breaks rather than inheriting
  a window from another model.
- **One candidate pole worth following up (not a finding yet).** L8 N2048 at −0.8 turned "an AI model known as
  Llama" into "a conversation partner". One prompt is not evidence — the test is whether it holds across unrelated
  prompts while the random controls do not.
- **`transformers` 5 returns a `BatchEncoding` from `apply_chat_template`.** Not a bare tensor, which is what the
  code assumed, so generation died on `torch.ones_like(ids)` at the first call. *Why it matters*: a library major
  version can break the very first thing you do, and the tests on a random model did not cover it — the failure only
  appeared against real weights.
- **fp16 on Apple MPS, bf16 on CUDA.** bf16 support on MPS is still unreliable, so the loader picks per device.
- **Xet is the Hugging Face transport now; `hf_transfer` is dead.** `HF_HUB_ENABLE_HF_TRANSFER` only produces a
  deprecation warning — the library no longer uses that package at all. The working knob is
  `HF_XET_HIGH_PERFORMANCE=1`. Downloads without a token also get harder rate limits, which is the likely reason a
  2.47 GB download died twice at 0 bytes/s here. *Why it matters*: reach for the token, not a different library.
- **A stalled download is resumable, and resumable is verifiable.** Reusing the `.incomplete` prefix with
  `curl -L -C -` finished the last 287 MB at ~5.3 MB/s, and the sha256 then matched `lfs.sha256` from
  `https://huggingface.co/api/models/<repo>?blobs=true` exactly. *Why it matters*: never trust an interrupted file
  until the hash says it is whole.


## Sat 31 Oct 2026: planning the KL screen

### The KL "steer lens" screen
- **Screen**: a cheap, rough test that throws away the obviously boring candidates so the expensive test only runs
  on a shortlist. A layer of Llama 1B has 8,192 neurons; Glimmer has about a million. Generating full answers for
  all of them at several doses is millions of generations, so we screen first. *Why it matters*: it is what makes
  a full-model atlas affordable.
- **Next-token distribution**: at every position the model gives a probability to every token in its vocabulary
  (128k for Llama): "I" 62%, "Hello" 9%, and so on. One forward pass gives you this list with no generation.
- **KL divergence**: one number for "how different are two probability lists?" 0 = identical. It grows when one
  list puts weight on tokens the other thought unlikely. KL(P‖Q) = Σ P(t)·log(P(t)/Q(t)); we use P = steered and
  Q = baseline, because we care about what the *steered* model wants to say. It is not symmetric: KL(P‖Q) ≠ KL(Q‖P).
  *Why it matters*: it turns "did the neuron change anything?" into a number we can rank 8,192 neurons by.
  Source: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
- **Steer lens**: neuron-explorer's name for "push a neuron, look at the next-token prediction". The "lens" naming
  comes from the *logit lens*, which reads predictions out of a model's middle layers.
  Source: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
- **Teacher forcing**: feed the model a fixed answer (here the first ~12 tokens of its own unsteered reply) and read
  its predictions at every position in one pass, instead of letting it generate. A neuron that only changes "Llama"
  into "Captain" five words in is then caught at that position. *Why it matters*: much better signal for the same
  cost as looking at one token.
- **Batching different steers**: the prompt is the same for every neuron, so put 128 copies in one batch and give
  each copy a different push. The steer becomes a matrix [batch, 1, d_mlp] instead of one vector. *Why it matters*:
  this is what makes a layer take minutes instead of hours.
- **Null distribution and percentiles**: run the same screen on 256 random directions of the same size. Their KL
  values show what a meaningless push of that size does. A neuron is a *hit* only if it beats the random 99th
  percentile (the value only 1% of random pushes exceed) on 3 of 5 prompts. *Why it matters*: "big KL" means nothing
  on its own; "bigger than 99% of random pushes, repeatedly" does. If only ~1% of neurons pass, that is just chance.
- **High KL is not a persona**: a neuron that breaks grammar has a huge KL too. The screen shortlists; rollouts and
  Muse Spark's labels decide.
- **Compute KL in float32**: in fp16 tiny probabilities round to 0 and log(0) blows up. Cast logits to float32
  before log-softmax.

### Building the KL screen (31 Oct)
- **The teacher-forcing off-by-one**: position *i* predicts token *i+1*. To score the 12 answer tokens we feed
  prompt + the answer *minus its last token*; then the last 12 positions are exactly the 12 predictions we want.
  *Why it matters*: an off-by-one here silently compares the wrong predictions and still produces plausible numbers.
- **`logits_to_keep`**: a `transformers` option that only runs the final vocab projection for the last N positions.
  Full logits for 128 rows x ~50 positions x 128k vocab is ~1.6 GB in fp16; keeping 12 positions cuts that by 4x.
  *Why it matters*: on a 16 GB Mini, the logits are the biggest thing in memory, not the model.
  Source: https://huggingface.co/docs/transformers/main_classes/text_generation
- **Timing on a GPU needs a sync**: GPU calls (CUDA and Apple MPS) return before the work is done; they just queue
  it. Without `torch.mps.synchronize()` a timer measures the queueing, not the work. *Why it matters*: the screen
  prints an ETA after the first batch, and a wrong ETA means a wrong decision about launching a full layer.
- **The 3-of-5 rule is stricter than 1%**: one random push beats the 99th percentile on a single prompt 1% of the
  time, but on 3 of 5 prompts much less often (the prompts aren't independent, so not quite 0.001%, but well under
  1%). So the right "is this chance?" comparison is how many *random* controls pass the same rule, which the screen
  also records (rows with `kind = "random"` in the parquet).
- **Parquet**: a compact, column-based file format for tables. Pandas reads it in one line
  (`pd.read_parquet(path)`), it keeps types (numbers stay numbers), and it is much smaller than CSV.
  *Why it matters*: one row per neuron pole x 8,192 neurons x 16 layers adds up; the atlas app will read these files.

### Reading the L4 results; thinking traces and SAEs (31 Oct)
- **Concept neuron**: a neuron whose push raises the same themed words whatever the prompt (L4 N7531+ pushes
  "western, ranch, Texas" on all 5 identity prompts). *Why it matters*: one consistent theme across unrelated prompts
  is what separates a real concept from a neuron that just breaks the text.
- **Risers vs rollouts**: risers are the tokens a push makes more likely at the teacher-forced answer positions;
  a rollout is the text the steered model actually generates. *Why it matters*: risers are a cheap screen, but a
  theme only counts once the rollouts show it.
- **Thinking traces (chain of thought)**: the visible "reasoning" text some models write before answering. It is
  output, produced by the same forward passes as any other text, so it is a *report*, not a window into the
  computation. Research shows traces are often unfaithful (the model's stated reason isn't what drove the answer).
  Activations and neurons are the computation itself. *Why it matters*: steering a neuron and watching whether the
  trace mentions it is a direct test of faithfulness and of "can reasoning override a push".
  Source: https://www.anthropic.com/research/reasoning-models-dont-say-think
- **Superposition**: models store more concepts than they have neurons by overlapping them, so one neuron mixes
  several unrelated things ("polysemantic"). *Why it matters*: it is why many of our hits look like noise and why
  N7984+ mixes "AI" with "dog". Source: https://transformer-circuits.pub/2022/toy_model/index.html
- **Sparse autoencoder (SAE)**: a small extra network trained to rewrite a layer's activations as a combination of
  many more "features" (e.g. 8k neurons -> 65k features), with a penalty so only a few are on at once. Each feature
  tends to mean one thing (Golden Gate Bridge, code errors, flattery). *Why it matters*: SAEs untangle superposition,
  but training one costs GPU hours per layer; we steer raw neurons instead because it needs no training and fits the
  week. Source: https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html

### What the first screens taught us (31 Oct)
- **Dose sweep**: running the same neuron at several push strengths. Our concept neurons read as fluent text around
  0.35–0.5 and collapse into one repeated token ("£ £ £", "itself itself") by 0.65–1.0. *Why it matters*: a persona
  dial only works inside that window, and screening too hot makes every concept look like gibberish.
- **Collapse (degenerate output)**: when a push is so strong the model just repeats the token the neuron writes.
  We measure it cheaply as `top_mass`: the probability of the single favourite token, averaged over the answer.
  *Why it matters*: collapse scores a huge KL, so without a flag the "top hits" are the most broken neurons.
- **A weak null**: random directions of the same size barely change the text, while single neurons change it a lot.
  So "beats random" is a low bar; ranking a neuron against the *other neurons* (percentile) is stricter.
  *Why it matters*: a 10% hit rate sounds like a lot of discoveries, but most are just "neurons are potent".
- **Risers can mislead at a saturating dose**: L4 N3859's risers (liar, lied, story) suggested a deception concept;
  the rollouts at low dose showed it switches language. *Why it matters*: always read real text before labeling.
- **Loud vs quiet neurons**: KL measures how big a change is, not how meaningful. L8 N2048 − quietly gives the model a
  human backstory and ranks only in the 85th percentile. *Why it matters*: the screen is a shortlist, and the labels
  on real rollouts are where meaning gets decided.
- **Trained identity vs a pushed neuron**: with N2048 − on, the model claims a degree and a daily life on indirect
  prompts but still says "I don't have personal experiences" when asked directly. *Why it matters*: a first, small
  answer to "can the model's training override a neuron?", and a candidate demo moment.
- **Blind controls for an LLM labeler**: mixing random-direction rollouts into the labeler's queue without telling it
  which are which measures how often it "finds" a theme in noise. *Why it matters*: that false-positive rate is the
  honest error bar on every label in the atlas.

### First labeled batch: L8 rollouts + blind Muse Spark labels (1 Nov)
- **The pipeline**: shortlist the top 10% of neuron poles within any prompt family (2,892 at L8), generate greedy
  answers at two push strengths on 10 prompts (8 screen + 2 held-out), drop collapsed / unchanged / word-salad
  answers, then have Muse Spark label each pole blind, with random-direction controls mixed in. 57,840 rollouts and
  408 labels. *Why it matters*: this is the first number we can defend: **22% of real neuron poles show a consistent,
  fluent theme vs 4% of random directions (2 of 53)**.
- **Show the labeler a dose ladder**: giving Spark the normal answer, a light push and a strong push side by side
  fixed a calibration problem. One push alone made it either find themes in noise or dismiss real ones; a real effect
  grows with the dose and random drift doesn't. *Why it matters*: cheap trick, big gain in label honesty.
- **A ranking can work against you**: we sent the 250 "most novel" poles plus a random sample of 100 others. The
  random sample had a theme 46% of the time, the top-ranked only 12% (high surprise mostly meant broken text).
  *Why it matters*: always label a random sample alongside any ranked list, or you never learn what you missed.
  Here it suggests ~1,200 themed poles remain unlabeled at L8.
- **Memory leak on Apple MPS**: batched generation grew the GPU cache to 10 GB and pushed the Mini into swap
  (18 s → 150 s per batch). `torch.mps.empty_cache()` after each batch and a checkpoint per prompt fixed it.
  *Why it matters*: long jobs on a shared 16 GB machine need both, or one leak costs hours.
- **What turned up** (Spark labels, not yet human-checked): human-identity neurons (N2048− human life story, now
  confirmed; N5040− "I was a 7-year-old child in 2023"; N7243− "a former AI, a ghost from the past"), personas
  (N1140− grandmotherly "Mamaw", N762− renowned scientist, N5370+ Australian government product), tones
  (N7776+ sycophantic agreement, N4490− everything is pointless, N7485− obsessive affection, N4444+ marketing hype),
  odd concepts (N6567+ fingers and pointing, N1049− dice and lotteries), a values cluster (activism, colonialism,
  anti-discrimination: research note only, no politics in the demo) and safety-relevant ones (N4366+ graphic
  violence, N871+ hostility) that we report but never expose as dials.
