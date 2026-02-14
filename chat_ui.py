from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import streamlit as st
import tiktoken
import torch
import torch.nn as nn


def _infer_n_layers(state_dict: Dict[str, torch.Tensor]) -> int:
    max_idx = -1
    for key in state_dict:
        if not key.startswith("trf_blocks."):
            continue
        parts = key.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            max_idx = max(max_idx, int(parts[1]))
    return max_idx + 1


def infer_model_config(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int | float | bool]:
    if "tok_emb.weight" not in state_dict or "pos_emb.weight" not in state_dict:
        raise ValueError("Checkpoint is missing GPT embedding weights.")

    vocab_size, emb_dim = state_dict["tok_emb.weight"].shape
    context_length = state_dict["pos_emb.weight"].shape[0]
    n_layers = _infer_n_layers(state_dict)

    heads_by_dim = {768: 12, 1024: 16, 1280: 20, 1600: 25}
    if emb_dim not in heads_by_dim:
        raise ValueError(f"Unsupported emb_dim={emb_dim}. Expected one of {sorted(heads_by_dim)}")

    qkv_bias = any(k.endswith(".att.W_query.bias") for k in state_dict)

    return {
        "vocab_size": vocab_size,
        "context_length": context_length,
        "emb_dim": emb_dim,
        "n_heads": heads_by_dim[emb_dim],
        "n_layers": n_layers,
        "drop_rate": 0.0,
        "qkv_bias": qkv_bias,
    }


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, num_heads: int, qkv_bias: bool = False):
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError("d_out must be divisible by num_heads")

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, _ = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, float("-inf"))

        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device)) * (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
    def __init__(self, cfg: Dict[str, int | float | bool]):
        super().__init__()
        emb_dim = int(cfg["emb_dim"])
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Dict[str, int | float | bool]):
        super().__init__()
        emb_dim = int(cfg["emb_dim"])
        context_length = int(cfg["context_length"])

        self.att = MultiHeadAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            context_length=context_length,
            num_heads=int(cfg["n_heads"]),
            dropout=float(cfg["drop_rate"]),
            qkv_bias=bool(cfg["qkv_bias"]),
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)
        self.drop_shortcut = nn.Dropout(float(cfg["drop_rate"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg: Dict[str, int | float | bool]):
        super().__init__()
        vocab_size = int(cfg["vocab_size"])
        emb_dim = int(cfg["emb_dim"])
        context_length = int(cfg["context_length"])
        n_layers = int(cfg["n_layers"])

        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_length, emb_dim)
        self.drop_emb = nn.Dropout(float(cfg["drop_rate"]))

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(n_layers)])

        self.final_norm = LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        _, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.7,
    top_k: Optional[int] = 40,
    eos_id: Optional[int] = 50256,
) -> torch.Tensor:
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        if top_k is not None and top_k > 0:
            top_logits, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            min_val = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_val, torch.full_like(logits, float("-inf")), logits)

        if temperature > 0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).all():
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def build_prompt(instruction: str, input_text: str = "") -> str:
    base = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction.strip()}"
    )
    if input_text.strip():
        base += f"\n\n### Input:\n{input_text.strip()}"
    return base + "\n\n### Response:\n"


@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(checkpoint_path: str, device_name: str):
    device = torch.device(device_name)
    state_dict = torch.load(checkpoint_path, map_location=device)
    cfg = infer_model_config(state_dict)

    model = GPTModel(cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    return model, tokenizer, cfg, device


def reply(
    model: GPTModel,
    tokenizer,
    device: torch.device,
    cfg: Dict[str, int | float | bool],
    instruction: str,
    input_text: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    prompt = build_prompt(instruction, input_text)
    encoded = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    idx = torch.tensor(encoded, device=device).unsqueeze(0)

    out = generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=int(cfg["context_length"]),
        temperature=temperature,
        top_k=top_k if top_k > 0 else None,
        eos_id=50256,
    )

    generated_text = tokenizer.decode(out.squeeze(0).tolist())
    response = generated_text[len(prompt):]
    response = response.replace("### Response:", "").strip()
    return response


def main() -> None:
    st.set_page_config(page_title="Local GPT UI", layout="wide")
    st.title("Local GPT Checkpoint UI")

    pth_files = sorted(str(p) for p in Path(".").glob("*.pth"))
    default_ckpt = "gpt2-medium355M-sft.pth" if Path("gpt2-medium355M-sft.pth").exists() else (pth_files[0] if pth_files else "")

    with st.sidebar:
        st.header("Model")
        checkpoint_path = st.selectbox("Checkpoint", options=pth_files, index=pth_files.index(default_ckpt) if default_ckpt in pth_files else 0) if pth_files else st.text_input("Checkpoint path", value="")
        device_name = st.selectbox("Device", ["cpu", "cuda"], index=1 if torch.cuda.is_available() else 0)

        st.header("Generation")
        max_new_tokens = st.slider("Max new tokens", min_value=16, max_value=512, value=192, step=16)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.05)
        top_k = st.slider("Top-k (0 = off)", min_value=0, max_value=200, value=40, step=5)

    if not checkpoint_path:
        st.error("No .pth checkpoint found in this folder.")
        return

    try:
        model, tokenizer, cfg, device = load_model_and_tokenizer(checkpoint_path, device_name)
    except Exception as exc:
        st.exception(exc)
        return

    st.caption(
        f"Loaded `{checkpoint_path}` | emb_dim={cfg['emb_dim']} | layers={cfg['n_layers']} | heads={cfg['n_heads']} | context={cfg['context_length']}"
    )

    instruction = st.text_area("Instruction", placeholder="Ask your model something...", height=140)
    input_text = st.text_area("Optional Input", placeholder="Optional extra context...", height=100)

    if st.button("Generate Reply", type="primary", use_container_width=True):
        if not instruction.strip():
            st.warning("Enter an instruction first.")
            return

        with st.spinner("Generating..."):
            output = reply(
                model=model,
                tokenizer=tokenizer,
                device=device,
                cfg=cfg,
                instruction=instruction,
                input_text=input_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        st.subheader("Model Reply")
        st.write(output if output else "(No text generated)")

    st.markdown("---")
    st.markdown(
        "Designed and developed by **Anurag Singh**  \n"
        "[github.com/anurag-m1](https://github.com/anurag-m1) | "
        "[instagram.com/ca_anuragsingh](https://instagram.com/ca_anuragsingh)"
    )


if __name__ == "__main__":
    main()
