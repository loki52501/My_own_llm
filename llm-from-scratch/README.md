# LLM from Scratch (PyTorch only)

This folder has two notebooks, and both use only PyTorch:

| Notebook | What it builds |
|---|---|
| [`LLM_From_Scratch.ipynb`](LLM_From_Scratch.ipynb) | a GPT that streams **every `.txt` file** (~30 GB) batch by batch: tokenizer → attention → resumable training → generation |
| [`Reasoning_Model_From_Scratch.ipynb`](Reasoning_Model_From_Scratch.ipynb) | a small **reasoning model** (`<think>…</think><answer>…</answer>`) built from the same books with pretraining → SFT → GRPO reinforcement learning → majority voting |

## Notebook 1: LLM from Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/loki52501/My_own_llm/blob/main/llm-from-scratch/LLM_From_Scratch.ipynb)

`LLM_From_Scratch.ipynb` builds and trains a small GPT on your own books. It uses only PyTorch, the Python standard library, and matplotlib for plots. Every section explains the step in plain language and includes a diagram.

| # | Section | What you build |
|---|---|---|
| 1 | Load the books | extract `txt-files.tar`, clean it, join the books with `<|endoftext|>` |
| 2 | Tokenizer | a character tokenizer, then a byte-level **BPE** tokenizer trained on your books |
| 3 | Dataset | a sliding-window `Dataset`/`DataLoader` where the target is the input shifted by one |
| 4 | Embeddings | token + positional embeddings |
| 5 | Self-attention | Q/K/V by hand, the causal mask, an attention heatmap, multi-head attention |
| 6 | GPT model | LayerNorm, GELU, MLP, residuals, the Transformer block, the full GPT |
| 7–8 | Training | AdamW, warm-up + cosine LR, gradient clipping, mixed precision, loss curves |
| 9 | Generation | temperature and top-k sampling, attention maps, embedding neighbours |
| 10 | Save / load | checkpoint + tokenizer (optionally to Google Drive) |

## Notebook 2: Reasoning Model from Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/loki52501/My_own_llm/blob/main/llm-from-scratch/Reasoning_Model_From_Scratch.ipynb)

This follows the DeepSeek-R1 / o1 recipe at toy scale:

| # | Stage | What happens |
|---|---|---|
| 1–2 | Books → problems | mine character names, countable objects and common words from your books |
| 3 | Verifiable tasks | story arithmetic ("Elizabeth has 7 letters…") and letter counting ("how many 'e' in 'netherfield'"), each with a worked chain of thought and an answer a program can check |
| 4 | Tokenizer | BPE with `<think> </think> <answer> </answer> <\|user\|> <\|assistant\|>` special tokens |
| 6 | Pretraining | the GPT learns English from the books |
| 7 | SFT ("cold start") | imitate worked solutions, with loss masking on the prompt |
| 8 | Evaluation | parse the `<answer>`, verify it, and compare **direct** answering against **thinking** |
| 9 | GRPO | reinforcement learning from scratch: group-relative advantages, clipped ratio, KL to a reference model, reward = correct answer + format |
| 10–11 | Results | accuracy per stage, plus self-consistency (majority vote) as test-time compute |

On a T4 GPU it takes about 20–30 minutes in total. It provides the books the same way as notebook 1 (see below).

## Running on Colab

**Data.** The data is the Hugging Face dataset [`Lokeshlks/gutenberg_books_text`](https://huggingface.co/datasets/Lokeshlks/gutenberg_books_text): `txt-files.tar`, about 30 GB. Notebook 1 streams **every `.txt` file** in the tar, batch by batch, from the first byte to the last:
- All languages are included, and no files are held out. Only the Gutenberg licence boilerplate is stripped.
- A background worker downloads and tokenizes the next batches while the GPU trains.
- Memory holds only the current file and a shuffle buffer; nothing is downloaded up front.

**Order:**
1. **`LLM_From_Scratch.ipynb` (T4 GPU):**
   - It trains until the stream reaches **the end of the tar**. The whole tar is ≈8B tokens, roughly 15–20 T4-hours.
   - Every 15 minutes it saves the model, the optimizer and the **byte position in the tar** to `MyDrive/llm_from_scratch/`.
   - When Colab disconnects, just **Run all** again: it resumes at that byte with an HTTP range request.
2. **`Reasoning_Model_From_Scratch.ipynb`:** it loads notebook 1's tokenizer and model from Drive as its pretrained base, then runs SFT → GRPO → majority voting (~20–30 min). If no base is found, it does a short streaming pretraining itself.

**Settings in notebook 1:**
- `SESSION_HOURS`: `None` means run to the end of the tar; a number stops early.
- `ENGLISH_ONLY`: `False` uses every file.
- `VAL_EVERY`: `None` trains on 100% of the files; `200` holds out 1 in 200.
- `CKPT_MINUTES`: how often to save.

It also runs locally on a CPU (`pip install torch matplotlib jupyter`), using a much smaller model.
