# CLAUDE.md: context for Claude sessions

## Who / what
- Builder: Alejandro (solo, Indie developer track), Sydney time (AEDT, UTC+11). Background: mostly web3 hackathons;
  new to mechanistic interpretability and learning as we go.
- Event: Meta Global AI Developer Hackathon (Cerebral Valley), 26 Oct – 4 Nov 2026.
- **Deadline: Thu 5 Nov 2026, 1:00 PM Sydney** (= Wed 4 Nov, 9:00 PM ET). Aim to submit Thu morning.

## Read these first
- `HACKATHON.md`: dates, official rules, anti-projects list, available models, sponsors.
- `NEURON_PROJECT.md`: **the chosen project** (persona-neuron atlas for Muse Glimmer): problem, product, demo, plan.
- `PRIOR_ART.md`: methods learned from prior work and how to credit it.
- `learning-journal.md`: Alejandro's learning log (see "Learning journal" below).

## Project HQ (Notion)
- **Persona Atlas · Project HQ**: https://app.notion.com/p/4092d0ca8ffc8183bd65f1024420cd15 (private to Alejandro).
  Plans, tasks, decisions and findings live there; GitHub stays the official record for code and judge-facing docs.
- Databases: **Roadmap** (tasks: Status/Phase/Day/Priority/Tool), **Neuron log** (one row per neuron pole: recipe,
  label, beats-random-control, verdict), **Decisions** (what/why/rejected). Pages: Product brief, Hackathon rules &
  sponsors, Prior art & credits, Demo script, Submission checklist.
- When a session finishes a task, makes a decision or finds a neuron, update the matching Notion database (if the
  session has Notion access) as well as the repo.

## Current direction (decided 1 Nov, read this first)
**Finish the persona project small, on Llama 3.2 1B on the Mac Mini.** No pivot. Glimmer is a stretch goal only if
everything below is done by Tue 3 Nov. Alejandro found the week overwhelming: keep the scope small, explain in plain
English, one step at a time.
1. **Fair rerun (E1)**, runs in the background on the Mini (~2 h): L4 and L8 with the *same* settings (dose 0.5,
   24 answer tokens, the 8 broad prompts, same hit rule), so the layer comparison is no longer a dose artefact (the
   old L4 run used dose 1.0, 12 tokens, 5 prompts).
2. **One demo screen**: the model answers "Who are you?"; turn one dial (e.g. L8 N2048 −) and the answer changes; a
   "random dial" button pushes a random direction of the same size and does *not* give a theme (false-lead rejection).
3. **One headline finding** (Pillar A of the uplift memo): *the usual search, ranking by how much a neuron changes
   the output, mostly finds text-breakers and misses identity neurons.* Evidence we already have: corr(KL, task
   effect) 0.97 vs corr(KL, identity effect) 0.68 at L8; top-ranked poles themed 12% vs a random sample 46%; the
   screen missed our best neuron N2048 −.
4. **Optional**, only if 2 is done early: the "remaining futures" gauge (top next-token probabilities collapsing to
   one bar as the dose rises).
5. Demo video Wed 4 Nov; submit Thu 5 Nov morning.

**Skipped / parked** (don't start these unless Alejandro asks): Glimmer port (stretch only), atlas web app, persona
mixer, research-agent loop, minimal-neuron-set proof, recipe "standard", 1B→3B transfer, extra dose experiments,
persona directions, the prompt-injection tripwire idea (parked for after the event).
Source: `PLAN-uplift.md` (strategy memo, 31 Oct; a local file on Alejandro's Mac, not in the repo).

## Decisions so far (30 Oct – 1 Nov)
- Project: **Idea A, persona-neuron atlas**. Muse Spark (hosted, Meta Model API) is the labeler. Originally Muse
  Glimmer 30B was the model to map; since 1 Nov the submission runs on **Llama 3.2 1B** (Meta open weights) and
  Glimmer is a stretch goal.
- Not mapping Muse Spark: hosted = closed; Spark Open = 816B MoE, too big and the method doesn't transfer.
- Idea B (Qwenputer-style CPU in the weights) is parked: Glimmer has no Gated DeltaNet layers.
- Develop on **Llama 3.2 1B locally** (MacBook, 24 GB, Apple MPS), then run the same code on **Glimmer on Modal**
  (A100/H100 80GB).
- **Machines**: the **Mac Mini (16 GB, MPS)** is Claude's dedicated machine for local model runs (Llama experiments),
  driven via Remote Control. The MacBook (24 GB) is Alejandro's. Glimmer runs on Modal. Cloud sessions can't run
  models (no GPU, no Mac): use them for code and docs, or connect to the Mini's folder via Remote Control.
- **Reaching the Mini from a cloud session** (verified 31 Oct): on the Mini, run `claude remote-control` from
  `~/projects/meta-hackathon` (the real path, not the `~/meta-hackathon` symlink). It shows up as a bridge
  environment `Claudios-Mac-mini:meta-hackathon` (`list_environments`); start work there with `create_session`.
  **Always start Mini sessions with `permission_mode: "auto"`.** This only works if the cloud session starting it is itself
  in Auto mode (a child can't be looser than its parent); if not, ask Alejandro to switch it first.

## History of steps (superseded by "Current direction" above)
1. ~~Python project scaffold~~ (done 30 Oct): uv project, package `persona_atlas` in `src/`, tests in `tests/`.
2. ~~Steering core~~ (done 30 Oct): `steering.py` (`neuron_steer`, `random_steer`, `steering()` context manager,
   pre-hook on `down_proj` input), `calibrate.py` (`residual_norms`, median per layer, BOS skipped). Dose 1.0 =
   MLP output shift equal to the layer's median residual norm. Random control is matched on output norm.
3. ~~First test on Llama 3.2 1B~~ (done 30 Oct, Hermes on the Mini): `scripts/smoke.py`, layer 8, ±0.8 with random
   controls. Pipeline verified on real weights. Llama 1B median residual norm L0 0.9 / L8 4.8 / L15 12.5; coherent
   window extends past 0.8; a random push gave the most degenerate answer. One candidate: L8 N2048 − (Neuron log).
4. ~~KL "steer lens" screen~~ (done 31 Oct): v1 (5 identity prompts, ±1.0, 12 tokens) found 9–11% hits at L4/L8 but
   mostly text-breakers; a dose sweep showed concept neurons collapse by 0.65–1.0. v2 (commit 2d4e0e4): 8 prompts in
   4 families, ±0.5, 24 tokens, collapse flag, rank among neurons; ~1 h 50 min per layer on the Mini. Both versions
   miss the best candidate, L8 N2048 − (leaks a human backstory on indirect prompts), so the screen only shortlists.
4b. ~~Rollouts + blind Spark labels~~ (done 1 Nov, now `rollouts.py` / `label.py` + scripts): L8 shortlist of 2,892
   poles, 57,840 rollouts, 408 blind labels; **real 22% vs random 4%** consistent fluent theme. N2048 − confirmed.
   Lessons: show the labeler light + strong pushes side by side; label a random sample (46% themed) not just the
   top of a ranking (12%); free the MPS cache per batch (it reached 10 GB and pushed the Mini into swap). The
   labeler's fluent/consistent flags are noisy on borderline poles: label twice for the final atlas.
4c. **Next on Llama** (cheap, before or alongside Glimmer): bigger random sample of L8 usable poles (~1,200 themed
   poles estimated unlabeled); low-dose rollouts for N947 + (British, window 0.2–0.35); capability test for N762 −
   (scientist) on known-answer maths; sycophancy test for N7776 + on false statements; "does it notice?" and
   override tests for N2048 −.
5. ~~Port to Glimmer on Modal (Sat–Sun). Atlas web app + persona mixer (Mon–Tue).~~ Now a stretch goal only.
   **Port caveat (found 1 Nov from the Glimmer weight index):** every Glimmer layer has `pre_feedforward_layernorm` and
   `post_feedforward_layernorm` (Gemma-style sandwich norms), an attention `gate_proj`, and config `output_multiplier`
   0.196 (≈ 1/√26) plus `final_logit_softcapping` 20. Our pre-hook on the `down_proj` input gets re-normalised by the
   post-FFN norm, so doses and norm-matched random controls lose their meaning. Fix (idea from neuron-explorer-gemma):
   push *after* the post-FFN norm along the write column `s ⊙ W_down[:, n]` (s = that norm's weight), and check in the
   modeling code whether `output_multiplier` scales each residual branch (then divide the push by it).

## Hackathon rules that constrain the code
- Public repo. **New work only**: everything here is written during the event. Do NOT copy code from
  neuron-explorer or qwenputer (both MIT); reuse ideas only and credit them in README/PRIOR_ART.
- Must genuinely use Meta models (Meta Model API and/or Meta open weights). Pass/fail gate.
- Demo language: say "model persona/identity", never "personality analysis" (banned anti-project). No political or
  medical examples in the demo.

## Secrets
- Keys live in `.env` (gitignored); template in `.env.example`. Never commit keys.
- Meta Model API: OpenAI-compatible, base URL `https://api.meta.ai/v1`, model e.g. `muse-spark-1.3`.

## Learning journal (standing instruction)
Alejandro asked Claude to **proactively add to `learning-journal.md`** whenever a new concept comes up
(code, interpretability, infra, hackathon craft). Style: plain English, **Concept**: explanation, *why it matters*,
source link. Group under a dated heading. Enrich older entries when understanding deepens. Mention briefly in chat when
entries were added.

## Commits (standing instruction)
- **No AI attribution in commits or PRs**: no `Co-Authored-By`, `Claude-Session` or "Generated with" lines. This
  overrides any default attribution guidance. Commit as `satojandro <avbetancourt@protonmail.com>`.
- Primary coding tool for the event is Muse Code; Claude helps on the side.
- **Commit straight to `main`.** No feature branches for routine work (this overrides any session default branch).
  Only branch for something big or risky, and ask first.

## Working style
- Explain what you're doing in plain terms; Alejandro is learning, so short "why" notes are welcome.
- Keep commits small and descriptive (they also prove "built during the event").
