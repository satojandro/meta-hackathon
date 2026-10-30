# Idea 1: Persona-neuron atlas for Muse Glimmer

Status: leading candidate for the Meta hackathon (30 Oct 2026). Worth returning to even if not picked.

## One-liner
A neuroscience lab for Meta's models: a Muse Spark agent maps the persona neurons inside Muse Glimmer, and you build
characters by turning neuron dials instead of writing prompts.

## The problem
- Giving a model a personality today means **system prompts** (fragile: jailbreakable, say nothing about *why* the
  model behaves as it does) or **fine-tuning** (expensive, slow, and you can't see what changed).
- Interpretability tooling exists for Qwen and Gemma, not for Meta's new Muse models.
- Neuron-finding is manual: the X researcher filtered ~295k neurons' outputs by hand; neuronary.org collects recipes
  one submission at a time.

## What we present (three layers)
1. **The atlas**: the first map of Muse Glimmer's persona neurons, by layer, each with a label, examples, and a
   random-control check. Headline finding, e.g. "Glimmer keeps its identity in layers X–Y, unlike Qwen."
2. **The research agent**: a Muse Spark agent given a goal ("find a lever that makes it a pirate"). It runs the
   experiments, rejects false leads that fail random controls, and returns a recipe.
3. **The persona mixer** (what mapping unlocks): build a custom character by stacking neuron dials
   (e.g. 40% sea captain + 20% sarcastic + 0% "I am an AI"). No prompt, no fine-tune. Export as a neuronary-style
   recipe anyone can load. Because the persona lives in the model's internals, prompt injection can't talk it out of
   character.

## Who it's for
- Game / companion app builders: characters that stay in character.
- Safety researchers: see the "I am an AI" and refusal levers directly.
- The interpretability community: Meta's model added to the shared maps.

## Fit with judging (Indie track)
- **Creativity**: personality as dials, not prompts.
- **Technical ambition**: sweeping ~1M neurons in a 30B model, an autonomous research agent, statistical controls.
- **Works**: live slider demo.
- **Theme**: Push the Frontier (understanding and controlling AI systems).

## Wording traps
- The anti-projects list bans "Personality Analyzers". Always say *model persona* / *model identity*, never
  "personality analysis". We edit the model; we don't analyze people.
- Keep political and medical examples out of the demo (the original Qwen work found a "president neuron").
  Pirates, aliens and football clubs are safe.

## 3-minute demo outline
1. Ask Glimmer "Who are you?" and get the normal answer.
2. Drag one slider: it becomes an alien.
3. The agent finds that neuron live, including rejecting a false lead that failed random controls.
4. Fly through the atlas.
5. Mix a custom character; show it ignores "ignore your instructions".
6. Close on the layer finding.

## Build plan
- **Fri**: steering pipeline, tested on Llama 3.2 1B on the MacBook (24 GB).
- **Sat–Sun**: Glimmer on Modal (A100/H100 80GB). KL screen of all neurons → full rollouts for top candidates →
  Spark agent labels them against random-direction controls.
- **Mon–Tue**: atlas web app + persona mixer.
- **Wed**: record demo. **Thu 5 Nov 1 PM Sydney**: submit.
- **Bonus (far-fetched, only if time)**: train a persona LoRA on Glimmer (Fireworks, free until 30 Nov), re-map, and see
  which neurons the fine-tune uses. A different approach; not needed for the submission, and also listed as future work.

## Sponsor usage
| Sponsor | Use |
|---|---|
| Meta Model API ($150) | Muse Spark as the research agent and labeler |
| Modal ($100) | GPU for the Glimmer sweep |
| Fireworks ($100) | Cheap hosted Glimmer/Spark Open for bulk labeling |
| Baseten ($100) | Backup host |

See PRIOR_ART.md for methods and credits.
