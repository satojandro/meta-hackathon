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

## Decisions so far (30 Oct)
- Project: **Idea A, persona-neuron atlas**. Muse Spark (hosted, Meta Model API) is the research agent and labeler;
  **Muse Glimmer 30B** (open weights, dense, SwiGLU, Apache 2.0) is the model we map.
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
  **Always start Mini sessions with `permission_mode: "auto"`.**

## Next steps (in order)
1. ~~Python project scaffold~~ (done 30 Oct): uv project, package `persona_atlas` in `src/`, tests in `tests/`.
2. ~~Steering core~~ (done 30 Oct): `steering.py` (`neuron_steer`, `random_steer`, `steering()` context manager,
   pre-hook on `down_proj` input), `calibrate.py` (`residual_norms`, median per layer, BOS skipped). Dose 1.0 =
   MLP output shift equal to the layer's median residual norm. Random control is matched on output norm.
3. ~~First test on Llama 3.2 1B~~ (done 30 Oct, Hermes on the Mini): `scripts/smoke.py`, layer 8, ±0.8 with random
   controls. Pipeline verified on real weights. Llama 1B median residual norm L0 0.9 / L8 4.8 / L15 12.5; coherent
   window extends past 0.8; a random push gave the most degenerate answer. One candidate: L8 N2048 − (Neuron log).
4. **KL "steer lens" screen** (next, field manual in the Notion Roadmap task): one batched forward pass per neuron,
   teacher-forced KL over the first ~12 baseline answer tokens, ±1.0, 5 identity prompts, vs 256 random controls;
   hit = beats the random 99th percentile on 3 of 5 prompts. Start at L4, then L2–L10. Then rollouts for top hits
   on mixed prompts → Muse Spark labels with a blind check vs controls.
5. Port to Glimmer on Modal (Sat–Sun). Atlas web app + persona mixer (Mon–Tue). Demo video (Wed).

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
