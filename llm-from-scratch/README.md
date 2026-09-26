# LLM from Scratch (PyTorch only)

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

## Running on Colab

1. Click the badge above. Then choose `Runtime → Change runtime type → T4 GPU`.
2. Provide your books (`txt-files.tar`, a tar of `.txt` files) in **one** of these ways:
   - **Upload** — run the notebook and an upload button appears when the tar isn't found.
   - **Google Drive** — put it at `MyDrive/llm_from_scratch/txt-files.tar` and set `USE_GOOGLE_DRIVE = True`.
   - **This repo** — commit it as `llm-from-scratch/data/txt-files.tar` (100 MB max per file on GitHub). The notebook downloads it from the raw GitHub URL.
   - Nothing provided? The notebook downloads a few public-domain Gutenberg books instead.
3. `Runtime → Run all`. The default GPU model has about 11M parameters and trains in roughly 10–20 minutes on a T4.

## Adding your tar file to the repo (from Windows)

```bash
git clone https://github.com/loki52501/My_own_llm.git
cd My_own_llm
copy F:\llm_from_scratch\txt-files.tar llm-from-scratch\data\
git add llm-from-scratch/data/txt-files.tar
git commit -m "Add training books"
git push
```

If the file is over 100 MB, use Google Drive instead, or [Git LFS](https://git-lfs.com/).

It also runs locally on a CPU (`pip install torch matplotlib jupyter`), using a much smaller model.
