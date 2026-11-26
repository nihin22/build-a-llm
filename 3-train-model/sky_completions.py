"""Load the trained tiny model and complete "The sky is" 100 times.

Usage (from this folder):  python sky_completions.py
Writes results/sky_completions.txt. Stops with an explanation if the beginning contains
tokens that never occurred in the training corpus, because the model has no internal ID for them.
"""

import sys
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F

from train import TinyGPT, BLOCK

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
BEGINNING = "The sky is"
N_COMPLETIONS = 100
NEW_TOKENS = 20

enc = tiktoken.get_encoding("o200k_base")
checkpoint = torch.load(HERE / "results" / "tiny_gpt.pt")
to_original = checkpoint["to_original"]
to_internal = {t: i for i, t in enumerate(to_original)}
model = TinyGPT(len(to_original))
model.load_state_dict(checkpoint["model_state"])
model.eval()

tokens = enc.encode_ordinary(BEGINNING)
print(f"{BEGINNING!r} -> GPT-4o tokens {[enc.decode([t]) for t in tokens]} = {tokens}")
missing = [enc.decode([t]) for t in tokens if t not in to_internal]
if missing:
    raise SystemExit(f"Cannot continue: token(s) {missing} never occur in the training corpus, "
                     f"so the model's {len(to_original):,}-token vocabulary has no ID for them.")

lines = [f"Beginning: {BEGINNING!r}, {N_COMPLETIONS} completions of {NEW_TOKENS} tokens, seeds 0-{N_COMPLETIONS - 1}\n"]
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

(HERE / "results" / "sky_completions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\nSaved results/sky_completions.txt")
