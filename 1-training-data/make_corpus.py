"""Extract the text field from each of the 10 FineWeb documents, in their original order,
replace line breaks with spaces and join the documents with a single space.
No summarising, rewriting, headings, metadata or separators.

Usage:  python make_corpus.py   (reads fineweb_sample.json, writes fineweb_corpus.txt in this folder)
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

data = json.loads((HERE / "fineweb_sample.json").read_text(encoding="utf-8"))
texts = [row["row"]["text"] for row in data["rows"]]
corpus = " ".join(re.sub(r"\r\n|\r|\n", " ", t) for t in texts)
(HERE / "fineweb_corpus.txt").write_text(corpus, encoding="utf-8")

print(f"{len(texts)} documents, {len(corpus):,} characters written to fineweb_corpus.txt")
