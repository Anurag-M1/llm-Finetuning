# Local GPT Checkpoint UI

A Streamlit web UI to chat with local GPT checkpoints generated from your notebook.

## Project Files

- `chat_ui.py` - Streamlit app for loading `.pth` checkpoints and generating replies
- `requirements-ui.txt` - Python dependencies
- `gpt2-medium355M-sft.pth` - fine-tuned checkpoint (from notebook)
- `model.pth` - GPT 124M model state dict checkpoint
- `llm & Finetuning.ipynb` - training and fine-tuning notebook

## Features

- Auto-detects model config from checkpoint (`emb_dim`, `layers`, `heads`, `context_length`)
- Supports local `.pth` selection from the UI
- Instruction + optional input prompting format
- Adjustable generation controls (`max_new_tokens`, `temperature`, `top_k`)

## Setup

Use Python 3.12 for better PyTorch compatibility on macOS.

```bash
cd '/Users/anurag/Desktop/llm & Finetuning'
'/opt/homebrew/bin/python3.12' -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-ui.txt
```

## Run

```bash
cd '/Users/anurag/Desktop/llm & Finetuning'
source .venv/bin/activate
streamlit run chat_ui.py
```

Open: [http://localhost:8501](http://localhost:8501)

## Credits

Designed and developed by Anurag Singh  
GitHub: [github.com/anurag-m1](https://github.com/anurag-m1)  
Instagram: [instagram.com/ca_anuragsingh](https://instagram.com/ca_anuragsingh)
