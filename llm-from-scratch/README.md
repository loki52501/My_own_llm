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

**Data.** The data is the Hugging Face dataset [`Lokeshlks/gutenberg_books_text`](https://huggingface.co/datasets/Lokeshlks/gutenberg_books_text): `txt-files.tar`, about 30 GB. Google Drive is not needed.

**Notebook 1 (`LLM_From_Scratch.ipynb`, Colab Pro GPU):**
1. **Section 3b** streams the **entire tar**, 256 files per batch. It tokenizes each batch in parallel on all CPU cores and appends the token ids to `llm_work/tokens.bin` on the runtime's local disk (≈16 GB).
   - Every `.txt` file is included, in all languages. Only the Gutenberg licence boilerplate is stripped.
   - It prints progress with an ETA, then the final totals: files, GB and tokens.
   - It takes about 1–2 hours and is resumable within the runtime.
2. **Section 8** trains on the whole `tokens.bin` (memory-mapped). One pass visits **every window exactly once**, in shuffled order, batch by batch. Progress shows as "% of corpus".
3. **Section 10** downloads `gpt_from_scratch.pt` + `tokenizer.json` to your computer.

**Notebook 2 (`Reasoning_Model_From_Scratch.ipynb`):** section 4 has an **upload button** for those two files. With them, notebook 1's model is the pretrained base for SFT → GRPO → majority voting. If you cancel the upload, it does a short streaming pretraining instead.

It also runs locally on a CPU (`pip install torch matplotlib jupyter`), using a much smaller model.
