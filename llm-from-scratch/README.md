# LLM from Scratch (PyTorch only)

This folder has two notebooks, and both use only PyTorch:

| Notebook | What it builds |
|---|---|
| [`LLM_From_Scratch.ipynb`](LLM_From_Scratch.ipynb) | a GPT trained on **every `.txt` file** of the ~30 GB tar (streamed batch by batch into a token file, then one full pass): tokenizer → attention → resumable training → generation |
| [`Reasoning_Model_From_Scratch.ipynb`](Reasoning_Model_From_Scratch.ipynb) | a small **reasoning model** (`<think>…</think><answer>…</answer>`) built from the same books with pretraining → SFT → GRPO reinforcement learning → majority voting |

## Notebook 1: LLM from Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/loki52501/My_own_llm/blob/main/llm-from-scratch/LLM_From_Scratch.ipynb)

`LLM_From_Scratch.ipynb` builds and trains a small GPT on your own books. It uses only PyTorch, the Python standard library, and matplotlib for plots. Every section explains the step in plain language and includes a diagram.

| # | Section | What you build |
|---|---|---|
| 1 | Load the books | extract `txt-files.tar`, clean it, join the books with `<|endoftext|>` |
| 2 | Tokenizer | a character tokenizer, then a byte-level **BPE** tokenizer trained on your books |
| 3 | Dataset | a sliding-window `Dataset`, then the **whole tar** streamed batch by batch into `tokens.bin` (every `.txt` file) |
| 4 | Embeddings | token + positional embeddings |
| 5 | Self-attention | Q/K/V by hand, the causal mask, an attention heatmap, multi-head attention |
| 6 | GPT model | LayerNorm, GELU, MLP, residuals, the Transformer block, the full GPT |
| 7–8 | Training | AdamW, warm-up + cosine LR, gradient clipping, mixed precision, loss curves |
| 9 | Generation | temperature and top-k sampling, attention maps, embedding neighbours |
| 10 | Save / download | model + tokenizer, downloaded for the reasoning notebook |

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

**Data.** The data is the Hugging Face dataset [`Lokeshlks/gutenberg_books_text`](https://huggingface.co/datasets/Lokeshlks/gutenberg_books_text): `txt-files.tar`, about 30 GB. Every `.txt` file is used.

**Survives reloads and disconnects.** The long work runs in `run.py`, a **background process** on the Colab machine, not in a notebook cell. Page reloads, closed tabs and kernel crashes don't stop it. Progress is saved to a **private Hugging Face repo** (`<you>/llm-from-scratch-run`), so a new runtime continues where the old one stopped. Google Drive isn't used.

**One-time setup.** Create a Hugging Face token with *write* access, then add it in Colab under 🔑 **Secrets** as `HF_TOKEN`, with notebook access enabled.

**Notebook 1 (`LLM_From_Scratch.ipynb`):**
1. **Run all.** Sections 1–7 build and explain everything. The tokenizer is trained once and saved to HF.
2. **Section 8, launch cell:** starts `run.py`, which
   - **prepares** the data: it streams every `.txt` file of the tar (256 files per batch), tokenizes on all CPU cores and uploads 1 GB token shards plus `progress.json` to HF;
   - **trains** on every window of every shard exactly once, checkpointing to HF every `CKPT_MINUTES`.
3. **Monitor cell:** shows the live log. Re-run it any time; stopping it doesn't stop training.
4. **If the runtime is recycled:** Run all again. The shards and checkpoint are downloaded and the job continues.
5. **When it says TRAINING DONE:** load the weights (end of section 8), then generate text (section 9).

**Notebook 2 (`Reasoning_Model_From_Scratch.ipynb`):** it downloads notebook 1's model from the same HF repo automatically (or shows an upload button), then runs SFT → GRPO → majority voting.

It also runs locally on a CPU (`pip install torch matplotlib jupyter`), using a much smaller model.
