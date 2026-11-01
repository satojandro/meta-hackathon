# Meta Global AI Developer Hackathon: brief

**Status:** Accepted, Indie developer track, idea not yet chosen (as of 30 Oct 2026).

## Key dates (Sydney time, AEDT)
- Submission deadline: **Thu 5 Nov, 1:00 PM** (= Wed 4 Nov, 9:00 PM ET)
- Judging: 5–8 Nov; closing ceremony and live finals: 9 Nov ET (10 Nov Sydney)
- Workshops and Discord office hours run until 4 Nov

## Indie track judging
Creativity, technical ambition, and whether it works.

## Themes (pick one)
- **Build for Your World:** bring Meta's models into a domain you know firsthand.
- **Push the Frontier:** software, robotics, manufacturing, energy, materials, science.

## Official rules (confirmed from participant portal)
- Public repo required
- Max 4 per team; solo allowed
- Demo must only show features/code built during the hackathon; judges must clearly see what's new, or it's disqualified at their discretion
- New work only (startups may build on existing code but must mark what's new)
- Banned: anything violating legal/ethical/platform policies, or using code/data/assets you don't have rights to
- Still unverified (from other participants): ~3-min demo video as main artifact; API credits must be claimed by submission day

## Anti-projects (avoid: generic or policy-violating)
- AI giving medical advice (mental health advisor, skin blemish detector, etc.)
- Basic RAG apps
- Simple image analyzers
- Simple "AI for Education" chatbots
- AI job application screener
- AI nutrition coach
- Personality analyzers
- Sports analyzers or coaches
- Anything political or involving war

## Models available
- **Muse Spark 1.x** (hosted): text/image/video/audio/PDF in, text out; 1M context; tool calling; structured output; reasoning_effort
- **Muse Wonder** (hosted): agentic coding, long-running agents, computer use
- **Muse Image**, **SAM 3.1** (segmentation), **Muse Voice Transcribe**
- **Muse Glimmer** (Apache 2.0 open weights, runs locally), Llama 4
- **Muse Code**: coding agent
- API: OpenAI-compatible, base URL `https://api.meta.ai/v1`; keys at dev.meta.ai
- Partner credits: Baseten, Modal, Fireworks ($100 each, select participants)

## Candidate ideas (current)
- **A. Persona-neuron atlas for Muse Glimmer**: leading candidate. Full write-up in NEURON_PROJECT.md.
- **B. Compute in latent space (Qwenputer-style CPU in the weights)**: fascinating, but not feasible on Meta's
  models this week (Glimmer has no Gated DeltaNet layers to use as RAM; the original is ~200k lines). See
  learning-journal.md. Parked for later.

## Earlier brainstorm (superseded)
1. Video → how-to guide (Spark video understanding + SAM key frames)
2. ~~Home hazard walk-through~~ (too close to "simple image analyzer")
3. Private local assistant on Glimmer, escalating hard cases to Spark
4. Lab notebook agent (Push the Frontier): logs observations from experiment footage

## Plan
*Updated 1 Nov:* persona project, small scope, Llama 3.2 1B on the Mini (see CLAUDE.md "Current direction").
Sun 1 Nov: fair L4/L8 rerun + start the demo screen. Mon–Tue: demo screen + headline finding (+ gauge if time,
Glimmer only if everything is done). Wed: video. Thu morning: submit.

Original plan:
- Fri 30 Oct: lock idea
- Sun 1 Nov: rough version working end to end
- Mon–Tue: polish + one "wow" feature
- Wed 4 Nov: record demo video
- Thu 5 Nov morning: submit with hours to spare
