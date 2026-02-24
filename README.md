# LLM Creation From Scratch and Fine-Tuning

This project covers the full workflow:

1. Build core LLM/GPT components from scratch (tokenization, embeddings, attention, transformer blocks, generation).
2. Fine-tune GPT-2 style models for instruction following.
3. Run a local Streamlit UI to chat with your saved `.pth` checkpoints.
   
<img width="1297" height="681" alt="llm   Finetuning" src="https://github.com/user-attachments/assets/b74197fb-2d1a-4cac-a4d7-1e20220f6a6c" />

## Repository Contents

- `llm & Finetuning.ipynb` - main notebook for model building, training, fine-tuning, and checkpoint export
- `chat_ui.py` - Streamlit app to load checkpoints and generate responses
- `requirements-ui.txt` - dependencies for UI/inference
- `the-verdict.txt` - local text corpus used in early training experiments
- `model.pth` - saved GPT 124M model state dict (from notebook step)
- `gpt2-medium355M-sft.pth` - fine-tuned GPT-2 medium checkpoint

## What Is Implemented

- GPT architecture blocks implemented in PyTorch:
  - LayerNorm
  - GELU/FeedForward
  - Causal Multi-Head Self-Attention
  - TransformerBlock
  - GPTModel
- Autoregressive text generation with:
  - temperature sampling
  - top-k filtering
  - EOS handling
- Instruction tuning workflow and checkpoint saving
- Local UI for interactive prompting and response generation

## Streamlit UI Features

- Auto-detect model config from checkpoint weights (`emb_dim`, `n_layers`, `n_heads`, `context_length`)
- Select local `.pth` checkpoint directly in sidebar
- Instruction + optional input format
- Adjustable decoding controls (`max_new_tokens`, `temperature`, `top_k`)

## Setup

Use Python 3.12 (recommended on macOS):

```bash
cd '/Users/anurag/Desktop/llm & Finetuning'
'/opt/homebrew/bin/python3.12' -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-ui.txt
```

## Run UI

```bash
cd '/Users/anurag/Desktop/llm & Finetuning'
source .venv/bin/activate
streamlit run chat_ui.py
```

Open:[http://localhost:8501]
live:(https://llm-finetuning-ykwu.onrender.com)

## Deploy On Render

1. Push this repository to GitHub.
2. In Render, click **New +** -> **Blueprint**.
3. Select the repo. Render will detect `render.yaml` and create the web service.
4. Deploy.

Render uses:
- Build command: `pip install --upgrade pip && pip install -r requirements-ui.txt`
- Start command: `streamlit run chat_ui.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`

Important:
- Keep at least one `.pth` checkpoint in the project root (or update the app to download/load one), otherwise the UI will start but cannot run inference.

## Credits

Designed and developed by Anurag Singh  
GitHub: [github.com/anurag-m1](https://github.com/anurag-m1)  
Instagram: [instagram.com/ca_anuragsingh](https://instagram.com/ca_anuragsingh)
