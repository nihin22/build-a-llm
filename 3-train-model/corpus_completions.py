"""Complete a beginning taken from the training corpus 100 times.

Usage (from this folder):  python corpus_completions.py

Picks a short sentence that appears exactly in fineweb_corpus.txt, preferring one with a common
opening ("The", "This", "It", ...). Its first few words become the beginning. Every token of the
beginning must exist in the trained model's saved vocabulary; otherwise the next sentence is tried.
Writes the chosen sentence, the beginning and 100 completions to results/corpus_completions.txt.
"""

import re
import sys
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F

from train import TinyGPT, BLOCK

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent / "1-training-data" / "fineweb_corpus.txt"
COMMON_OPENINGS = ["The", "This", "It", "We", "In", "A", "There"]
BEGINNING_WORDS = 3
N_COMPLETIONS = 100
NEW_TOKENS = 20

enc = tiktoken.get_encoding("o200k_base")
checkpoint = torch.load(HERE / "results" / "tiny_gpt.pt")
to_original = checkpoint["to_original"]
to_internal = {t: i for i, t in enumerate(to_original)}
model = TinyGPT(len(to_original))
model.load_state_dict(checkpoint["model_state"])
model.eval()

corpus = CORPUS.read_text(encoding="utf-8")
sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", corpus)]
short = [s for s in sentences if 5 <= len(s.split()) <= 15 and s[0].isupper()]
# Sentences with a common opening first, in corpus order; then the rest.
candidates = [s for s in short if s.split()[0] in COMMON_OPENINGS] + [s for s in short if s.split()[0] not in COMMON_OPENINGS]

for sentence in candidates:
    beginning = " ".join(sentence.split()[:BEGINNING_WORDS])
    tokens = enc.encode_ordinary(beginning)
    if sentence in corpus and all(t in to_internal for t in tokens):
        break
    print(f"Skipped {beginning!r}: token(s) not in the model's vocabulary")
else:
    raise SystemExit("No suitable sentence found")

print(f"Sentence:  {sentence!r}")
print(f"Beginning: {beginning!r} -> tokens {[enc.decode([t]) for t in tokens]}\n")

lines = [f"Chosen sentence (appears exactly in fineweb_corpus.txt): {sentence}",
         f"Beginning: {beginning!r}",
         f"{N_COMPLETIONS} completions of {NEW_TOKENS} tokens, seeds 0-{N_COMPLETIONS - 1}", ""]
with torch.no_grad():
    for seed in range(N_COMPLETIONS):
        g = torch.Generator().manual_seed(seed)
        idx = torch.tensor([[to_internal[t] for t in tokens]])
        for _ in range(NEW_TOKENS):
            probs = F.softmax(model(idx[:, -BLOCK:])[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1, generator=g)], dim=1)
        text = enc.decode([to_original[i] for i in idx[0].tolist()])
        lines.append(f"{seed + 1:>3}. {text}")
        print(lines[-1])

(HERE / "results" / "corpus_completions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\nSaved results/corpus_completions.txt")
