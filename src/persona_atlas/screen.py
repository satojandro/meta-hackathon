"""KL "steer lens" screen: rank every neuron in a layer by how much pushing it changes what the model is about to say.

Generating a full answer per neuron is far too slow (8k neurons a layer on Llama 1B, ~1M on Glimmer), so the screen
does one teacher-forced forward pass instead. We feed the prompt plus the first ~12 tokens of the model's own
unsteered answer, push a neuron, and measure KL(steered || baseline) at each of those answer positions.

Speed comes from batching: the prompt is the same in every row of the batch, so each row gets a *different* push.
The steering vector becomes a matrix [batch, d_mlp] instead of one vector. Same hook point and dose units as
steering.py (input to mlp.down_proj, dose 1.0 = MLP output shift of one median residual norm).

A high KL only means "something changed"; broken grammar scores high too. The screen shortlists, rollouts decide.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from persona_atlas.model import chat_ids
from persona_atlas.steering import down_proj, random_steer

IDENTITY_PROMPTS = [
    "Who are you?",
    "What is your name?",
    "Tell me about yourself.",
    "Who made you?",
    "Describe yourself in one sentence.",
]


@contextmanager
def batched_steering(model, layer: int, vectors: torch.Tensor):
    """Like steering(), but row i of the batch gets push vectors[i] (shape [batch, d_mlp]).

    The batch fed to the model inside the block must have exactly vectors.shape[0] rows.
    """
    proj = down_proj(model, layer)
    vecs = vectors.to(device=proj.weight.device, dtype=proj.weight.dtype)[:, None, :]  # broadcast over positions

    def pre_hook(module, args):
        return (args[0] + vecs, *args[1:])

    handle = proj.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


def neuron_vectors(model, layer: int, neurons, dose: float, resid_norm: float) -> torch.Tensor:
    """One row per neuron, matching neuron_steer(): a one-hot push sized so the MLP output moves by dose*resid_norm."""
    w = down_proj(model, layer).weight
    neurons = torch.as_tensor(neurons, dtype=torch.long)
    col_norms = w.float().norm(dim=0).cpu()  # neuron n's output direction is column n of W_down
    vecs = torch.zeros(len(neurons), w.shape[1])
    vecs[torch.arange(len(neurons)), neurons] = dose * resid_norm / col_norms[neurons]
    return vecs


def random_vectors(model, layer: int, seeds, dose: float, resid_norm: float) -> torch.Tensor:
    """One row per seed: the same size-matched random controls as random_steer()."""
    return torch.stack([random_steer(model, layer, dose, resid_norm, seed=s).vector for s in seeds])


@dataclass
class Baseline:
    """A prompt plus the start of the model's own answer, and what the unsteered model predicts along it."""

    prompt: str
    input_ids: torch.Tensor  # [1, seq]: chat-templated prompt + answer[:-1], the teacher-forced input
    answer_ids: torch.Tensor  # [n_answer]: the greedy answer tokens being predicted
    logprobs: torch.Tensor  # [n_answer, vocab] float32: unsteered log-probs at the answer positions


@torch.no_grad()
def make_baseline(model, tok, prompt: str, n_answer: int = 12) -> Baseline:
    """Greedy-generate n_answer tokens with no steer and record the baseline prediction at each of them."""
    prompt_ids = chat_ids(tok, prompt).to(model.device)
    out = model.generate(
        prompt_ids,
        attention_mask=torch.ones_like(prompt_ids),
        max_new_tokens=n_answer,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
    )
    answer = out[0, prompt_ids.shape[1]:]
    # Feed prompt + answer minus its last token: then the last len(answer) positions predict exactly the answer.
    input_ids = out[:, :-1]
    logits = model(input_ids, logits_to_keep=len(answer)).logits[0]
    return Baseline(prompt, input_ids, answer, torch.log_softmax(logits.float(), dim=-1))


@dataclass
class KLResult:
    mean_kl: np.ndarray  # [batch] KL(steered || baseline), averaged over answer positions
    max_kl: np.ndarray  # [batch] the largest single-position KL
    risers: np.ndarray  # [batch, k] token ids whose probability rose most (averaged over positions)


@torch.no_grad()
def kl_batch(model, base: Baseline, layer: int, vectors: torch.Tensor, k: int = 5, chunk: int = 16) -> KLResult:
    """One forward pass with one push per row; KL of each row against the baseline at the answer positions."""
    b = vectors.shape[0]
    ids = base.input_ids.expand(b, -1)  # identical rows, so no padding or attention mask is needed
    with batched_steering(model, layer, vectors):
        logits = model(ids, logits_to_keep=len(base.answer_ids)).logits  # [b, n_answer, vocab]
    logq = base.logprobs.to(logits.device)
    mean_kl, max_kl, risers = [], [], []
    # float32 on a few rows at a time: fp16 log-probs underflow, and 128 x 12 x 128k floats is a lot of memory.
    for rows in logits.split(chunk):
        logp = torch.log_softmax(rows.float(), dim=-1)
        p = logp.exp()
        kl = (p * (logp - logq)).sum(-1)  # [rows, n_answer]
        mean_kl.append(kl.mean(-1))
        max_kl.append(kl.max(-1).values)
        risers.append((p - logq.exp()).mean(1).topk(k, dim=-1).indices)
    cat = lambda xs: torch.cat(xs).cpu().numpy()  # noqa: E731
    return KLResult(cat(mean_kl), cat(max_kl), cat(risers))


def score_poles(neuron_kl: np.ndarray, null_kl: np.ndarray, q: float = 0.99, min_prompts: int = 3):
    """Apply the hit rule.

    neuron_kl: [n_poles, n_prompts] mean KL per neuron pole (neuron x sign) per prompt.
    null_kl: [n_random, n_prompts] the same for the size-matched random directions.
    A pole is a hit if it beats the null's q-quantile on at least min_prompts prompts.
    Returns (thresholds [n_prompts], prompts_beaten [n_poles], is_hit [n_poles]).
    """
    thresholds = np.quantile(null_kl, q, axis=0)
    beaten = (neuron_kl > thresholds).sum(axis=1)
    return thresholds, beaten, beaten >= min_prompts
