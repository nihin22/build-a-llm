"""Train a tiny GPT-style model on fineweb_corpus.txt for about 60 seconds on CPU.

Usage (from this folder):  python train.py

1. Reads ../1-training-data/fineweb_corpus.txt and ../2-text-representation/fineweb_tokens.txt,
   checks that the token IDs match the corpus under GPT-4o's o200k_base tokenizer,
   and regenerates them from the corpus if they do not.
2. Remaps the token IDs that occur in the corpus to a small internal vocabulary
   (0, 1, 2, ...) so the model only needs an output for tokens it can actually see.
3. Trains a randomly initialised causal transformer to predict the next token.
   Prints  step | loss | elapsed time | tokens/second  while training.
4. Generates text from the same short beginning and random seed before training,
   every 20 updates and at the end, streaming the tokens as they are produced.
5. Saves to results/: training_log.csv, loss_plot.png, samples.txt, tiny_gpt.pt
"""

import csv
import re
import sys
import time
from pathlib import Path

import matplotlib
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent / "1-training-data" / "fineweb_corpus.txt"
TOKENS = HERE.parent / "2-text-representation" / "fineweb_tokens.txt"
RESULTS = HERE / "results"

SEED = 1337
TRAIN_SECONDS = 60
BLOCK = 64          # context length: how many previous tokens the model can look at
BATCH = 16          # sequences per update
EMBED = 128         # size of each token's vector
HEADS = 4
LAYERS = 2
LEARNING_RATE = 3e-3
PRINT_EVERY = 10
SAMPLE_EVERY = 20
SAMPLE_TOKENS = 30
# The first beginning whose tokens all occur in the corpus is used for every sample.
PROMPT_CANDIDATES = ["The", "A novel two-step", "Kraft Foods has", "|Viewing Single Post"]


def load_tokens(enc):
    corpus = CORPUS.read_text(encoding="utf-8")
    expected = enc.encode_ordinary(corpus)
    given = [int(x) for x in re.findall(r"\d+", TOKENS.read_text(encoding="utf-8-sig"))]
    print(f"Corpus: {len(corpus):,} characters -> {len(expected):,} tokens with o200k_base")
    print(f"fineweb_tokens.txt: {len(given):,} token IDs")
    if given == expected:
        print("The token file matches the corpus exactly.\n")
        return expected

    for start in range(len(given) - len(expected) + 1):
        if given[start:start + len(expected)] == expected:
            extra = given[:start] + given[start + len(expected):]
            names = []
            for t in extra:
                try:
                    names.append(enc.decode([t]))
                except KeyError:
                    names.append(f"<special {t}>")
            print(f"The corpus tokens appear unchanged inside the file, but {len(extra)} extra token(s) surround them:")
            print("  " + " ".join(repr(n) for n in names))
            print("  These are the chat-format wrapper Tiktokenizer adds (system message, user turn, assistant turn).")
            break
    else:
        print("The token file does NOT match the corpus.")
    print("Using token IDs regenerated from the corpus.\n")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "corpus_tokens_regenerated.txt").write_text(", ".join(map(str, expected)), encoding="utf-8")
    return expected


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED)
        self.attn = nn.MultiheadAttention(EMBED, HEADS, batch_first=True)
        self.ln2 = nn.LayerNorm(EMBED)
        self.mlp = nn.Sequential(nn.Linear(EMBED, 4 * EMBED), nn.GELU(), nn.Linear(4 * EMBED, EMBED))

    def forward(self, x, mask):
        a = self.ln1(x)
        x = x + self.attn(a, a, a, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, EMBED)
        self.pos_emb = nn.Embedding(BLOCK, EMBED)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.ln_f = nn.LayerNorm(EMBED)
        self.head = nn.Linear(EMBED, vocab_size)

    def forward(self, idx):
        t = idx.size(1)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)  # a token may not look ahead
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(t))
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln_f(x))


@torch.no_grad()
def generate(model, enc, prompt_ids, to_original, label):
    """Sample SAMPLE_TOKENS tokens after the prompt, printing each as it appears. Same seed every time."""
    model.eval()
    g = torch.Generator().manual_seed(SEED)
    idx = torch.tensor([prompt_ids])
    prompt_text = enc.decode([to_original[i] for i in prompt_ids])
    print(f"  [{label}] {prompt_text}", end="", flush=True)
    pieces = []
    for _ in range(SAMPLE_TOKENS):
        logits = model(idx[:, -BLOCK:])[:, -1, :]
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=g)
        idx = torch.cat([idx, nxt], dim=1)
        piece = enc.decode([to_original[nxt.item()]])
        pieces.append(piece)
        print(piece, end="", flush=True)
    print()
    model.train()
    return prompt_text + "".join(pieces)


def main():
    torch.manual_seed(SEED)
    enc = tiktoken.get_encoding("o200k_base")
    tokens = load_tokens(enc)

    to_original = sorted(set(tokens))                  # internal id -> GPT-4o token id
    to_internal = {t: i for i, t in enumerate(to_original)}
    data = torch.tensor([to_internal[t] for t in tokens])
    print(f"{len(to_original):,} distinct tokens in the corpus -> internal vocabulary 0..{len(to_original) - 1}")
    print(f"(GPT-4o's full vocabulary has {enc.n_vocab:,} tokens)\n")

    prompt = next(p for p in PROMPT_CANDIDATES if all(t in to_internal for t in enc.encode_ordinary(p)))
    prompt_ids = [to_internal[t] for t in enc.encode_ordinary(prompt)]

    model = TinyGPT(len(to_original))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {LAYERS} layers, {HEADS} heads, {EMBED}-dim embeddings, context {BLOCK} tokens, {n_params:,} parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    batch_gen = torch.Generator().manual_seed(SEED)

    samples = [("before training", 0, generate(model, enc, prompt_ids, to_original, "step 0, before training"))]
    print(f"\nTraining for about {TRAIN_SECONDS} seconds on CPU...")
    print(f"{'step':>6} | {'loss':>7} | {'elapsed':>8} | {'tokens/s':>9}")

    log = []
    step, train_time, start = 0, 0.0, time.time()
    while train_time < TRAIN_SECONDS:
        t0 = time.time()
        ix = torch.randint(len(data) - BLOCK - 1, (BATCH,), generator=batch_gen)
        x = torch.stack([data[i:i + BLOCK] for i in ix])
        y = torch.stack([data[i + 1:i + BLOCK + 1] for i in ix])   # the target is the next token
        loss = F.cross_entropy(model(x).reshape(-1, len(to_original)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_time += time.time() - t0
        step += 1

        tps = step * BATCH * BLOCK / train_time
        log.append({"step": step, "loss": round(loss.item(), 4), "elapsed_s": round(time.time() - start, 2),
                    "tokens_per_s": round(tps)})
        if step == 1 or step % PRINT_EVERY == 0:
            print(f"{step:>6} | {loss.item():>7.4f} | {time.time() - start:>7.1f}s | {tps:>9,.0f}")
        if step % SAMPLE_EVERY == 0:
            samples.append((f"step {step}", step, generate(model, enc, prompt_ids, to_original, f"step {step}")))

    print(f"{step:>6} | {loss.item():>7.4f} | {time.time() - start:>7.1f}s | {tps:>9,.0f}   (final)")
    samples.append(("after training", step, generate(model, enc, prompt_ids, to_original, f"step {step}, after training")))

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "training_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss", "elapsed_s", "tokens_per_s"])
        w.writeheader()
        w.writerows(log)
    with open(RESULTS / "samples.txt", "w", encoding="utf-8") as f:
        f.write(f"Beginning: {prompt!r}, seed {SEED}, {SAMPLE_TOKENS} tokens per sample\n\n")
        for label, s, text in samples:
            f.write(f"[{label}] {text}\n\n")

    plt.figure(figsize=(8, 4.5))
    plt.plot([r["step"] for r in log], [r["loss"] for r in log], linewidth=1)
    plt.xlabel("training step")
    plt.ylabel("cross-entropy loss")
    plt.title(f"Tiny GPT on {len(tokens):,} FineWeb tokens ({step} steps, ~{TRAIN_SECONDS}s CPU)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "loss_plot.png", dpi=120)

    torch.save({"model_state": model.state_dict(), "to_original": to_original,
                "config": {"block": BLOCK, "embed": EMBED, "heads": HEADS, "layers": LAYERS}},
               RESULTS / "tiny_gpt.pt")
    print(f"\nSaved training_log.csv, loss_plot.png, samples.txt and tiny_gpt.pt to {RESULTS}")


if __name__ == "__main__":
    main()
