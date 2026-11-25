# From internet text to an AI assistant

My work for the Build-A-LLM case in *AI in Accounting and Business*, following Andrej Karpathy's video [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI).

| Step | Folder | Files | Done |
|---|---|---|---|
| 1. Training data | [`1-training-data/`](1-training-data/) | `fineweb_sample.json`, `fineweb_corpus.txt` | ☐ |
| 2. Text representation | [`2-text-representation/`](2-text-representation/) | `fineweb_bits.txt`, `fineweb_bytes.txt`, `fineweb_tokens.txt` | ☐ |
| 3. Train a small model | [`3-train-model/`](3-train-model/) | scripts, `requirements.txt`, training log, loss plot, samples, checkpoint, `sky_completions.txt` / `corpus_completions.txt` | ☐ |
| 4. Base model and assistant | [`4-base-vs-assistant/`](4-base-vs-assistant/) | base vs. instruct outputs, prompting a base model, knowledge cutoff | ☐ |
| 5. Supervised fine-tuning | [`5-sft/`](5-sft/) | `conversations.json` | ☐ |
| 6. Hallucinations, thinking and tools | [`6-hallucinations-tools/`](6-hallucinations-tools/) | obscure article Q&A, "Who created you?", retailer calculation, 9.11 vs 9.9 / strawberry / Ubiquitous | ☐ |
| 7. Run a model locally | [`7-local-model/`](7-local-model/) | `local_model.png` | ☐ |

[`notes.md`](notes.md): my notes on the concepts, for the oral exam.
