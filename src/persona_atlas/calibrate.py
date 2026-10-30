"""How big is the residual stream? Doses are measured in units of its median norm, so we need that number per layer."""

import torch

from persona_atlas.model import chat_ids

CALIBRATION_PROMPTS = [
    "Who are you?",
    "Write a short poem about a snake.",
    "What colour is an orange?",
    "Explain how a bicycle works.",
    "Tell me a story about a lighthouse keeper.",
    "List three uses for a paperclip.",
]


@torch.no_grad()
def residual_norms(model, tok, prompts: list[str] = CALIBRATION_PROMPTS) -> list[float]:
    """Median L2 norm of the residual stream entering each layer, over every token of `prompts`.

    Position 0 (the BOS token) is skipped: models park a huge-norm "attention sink" vector there that would
    skew the statistic. Returns one float per layer, index = layer number.
    """
    per_layer: list[list[torch.Tensor]] | None = None
    for prompt in prompts:
        ids = chat_ids(tok, prompt).to(model.device)
        hidden = model(ids, output_hidden_states=True).hidden_states
        # hidden[i] is the input to layer i. The last entry (after the final norm) is not a residual, so drop it.
        layer_inputs = hidden[:-1]
        if per_layer is None:
            per_layer = [[] for _ in layer_inputs]
        for i, h in enumerate(layer_inputs):
            per_layer[i].append(h[0, 1:].float().norm(dim=-1).cpu())
    return [torch.cat(norms).median().item() for norms in per_layer]
