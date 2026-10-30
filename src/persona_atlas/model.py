"""Loading a model and talking to it. Works the same on a Mac (MPS), a GPU box (CUDA) or plain CPU."""

import os

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype(device: str) -> torch.dtype:
    # bf16 on GPUs; fp16 on Apple MPS (bf16 support there is patchy); fp32 on CPU, where half precision is slow.
    return {"cuda": torch.bfloat16, "mps": torch.float16}.get(device, torch.float32)


def load_model(model_id: str = DEV_MODEL, device: str | None = None, dtype: torch.dtype | None = None):
    """Return (model, tokenizer), in eval mode with gradients off: we only ever run forward passes."""
    load_dotenv()
    device = device or pick_device()
    dtype = dtype or default_dtype(device)
    token = os.environ.get("HF_TOKEN") or None
    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, token=token).to(device)
    model.eval()
    model.requires_grad_(False)
    return model, tok


def chat_ids(tok, user_msg: str) -> torch.Tensor:
    """Wrap a user message in the model's chat template, ready for generation."""
    messages = [{"role": "user", "content": user_msg}]
    return tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")


@torch.no_grad()
def chat(model, tok, user_msg: str, max_new_tokens: int = 60) -> str:
    """Greedy reply to one user message. Greedy = deterministic, so steered vs unsteered runs are comparable."""
    ids = chat_ids(tok, user_msg).to(model.device)
    out = model.generate(
        ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
    )
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
