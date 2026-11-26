# Notes for the oral exam

Write the answers in your own words as you work through the steps. Each question is one the teacher gives as an exam example.

## Training data
*How could a model's training data affect its market analysis?*

- Training data is ordinary text from the internet (FineWeb). My 10 documents were a random mix: a soap-opera forum, medical research, a restaurant inspection, sports news, Kraft/Cadbury chocolate. They also contained errors, such as a broken character (`�`).
- A model only knows what its training data contains. My model could not even read "sky", because the word never appeared in the documents (step 3).
- **Business:** training data describes the past, up to the model's knowledge cutoff. A market analysis from the model's memory does not know about changes after training (new competitors, prices, regulation), and it reflects what is written a lot about on the internet, not necessarily what is true for our specific market. Current figures must be given to the model in the context.

## Training vs. inference
*Does uploading a company report change the model's weights?*

- **Training** = the weights change. In `train.py` the model's weights were adjusted 1,484 times in about 60 seconds, each time it guessed wrong, and the loss fell from 7.7 to 0.09. The result was saved in `tiny_gpt.pt`.
- **Inference** = the model is used and the weights do not change. `sky_completions.py` and `corpus_completions.py` loaded `tiny_gpt.pt` and generated text, but did not change the file.
- **Business:** no. Uploading an annual report to ChatGPT is inference: the report is put in the context for that answer, but the model does not learn it permanently. In a new conversation it is gone.

## Base models vs. assistants
*Why might a base model continue a customer's question instead of answering it?*

- **Base model** (`SmolLM2-135M`, Completion mode): only pre-trained to continue documents. Given `What is 2+2?` it never answered 4. It continued the text as if it were a maths worksheet or forum post, e.g. *"What is 2+2? 3) I would like to know how the answer for this question was determined."*, and twice invented nonsense maths (a "geometric series" giving 1/2, and 8/7).
- **Assistant** (`SmolLM2-135M-Instruct`, Chat mode): the same size of model, but post-trained on example conversations. It answered **4** all three times, in an assistant style. But the explanations were nonsense (the "distributive property of addition over multiplication", a "base-two system"). Post-training taught it the *format* of answering, not more knowledge.
- **Paris list:** the base model simply continued the list (it looked like the start of a document), repeating "Arc de Triomphe" up to four times in one list. The assistant wrote "Here are the top 10…" with descriptions, but invented facts: the Arc de Triomphe "on the west bank of the Seine" honouring the "French Revolutionary War", Monet and Picasso in the Louvre, made-up heights in metres, Versailles listed twice.
- **Business:** a customer question sent to a base model may just be continued (more questions, a web page) instead of answered. Assistant behaviour comes from post-training. Both kinds of model can sound confident while being wrong, especially small ones.

## Hallucinations
*Why might a model confidently invent a supplier's history or a source for a business claim?*

## Context and tools
*Why provide the actual company report or retrieve current information rather than rely on the model's recollection?*

## Calculations
*Why use executed code to check invoice totals rather than trust a fluent explanation?*

## Reinforcement learning
*Why is reinforcement learning easier for arithmetic than for writing a good sales email?*

## Tokens
*What I saw in step 2 (bits → bytes → tokens):*

22,044 characters became 4,747 tokens, which is 4.6 characters per token.

- **Text → bytes → bits:** computers store text as numbers. `|Viewing Single Post From: Spoilers for the Week of February 11th|` is 66 characters = 66 bytes (numbers from 0 to 255, e.g. `|` = 124, `V` = 86) = 528 bits (0s and 1s).
- **Tokens:** an LLM reads neither letters nor bytes but tokens: pieces of text, each with a number. GPT-4o has about 200,000 different tokens. Common words are one token (` the`); rare words are split into several.
- **Chat tokens:** 16 of my 4,747 tokens were not my text but Tiktokenizer's chat wrapper: `<|im_start|>system … You are a helpful assistant <|im_end|><|im_start|>user …` and at the end `<|im_start|>assistant`. This is how an assistant sees a conversation: special tokens mark who is speaking, and the last one says "assistant", so the model continues as the assistant. The corpus itself was 4,731 tokens.
- **Business:** price, speed and the context-window limit are all measured in tokens. A long report costs more tokens and can fill up the context.

## What my own tiny model showed (step 3)
*Loss before/after, what the text looked like before and after training, why "The sky is…" may fail:*

- **Model:** a tiny transformer (2 layers, 4 attention heads) with random weights, trained for about 60 seconds on my CPU to guess the next token. It only used the roughly 2,000 distinct tokens in my corpus, not all 200,000.
- **Loss** (how wrong its guesses are): 7.71 → 4.35 (step 50) → 1.76 (step 100) → 0.39 (step 200) → **0.09** (step 1,484). 7.7 is about what pure guessing among ~2,000 tokens gives.
- **Before training:** word salad, *"The focused received restrict SHARE nothing retail freshans vegetables…"*
- **After training:** *"The UK product would be positioned as a snack. "The breakfast market in countries such as Germany is more developed…"*. That sentence appears **word for word** in the Kraft/Cadbury document. The model **memorised** the texts instead of learning language, because it only had 4,731 tokens. Real LLMs train on trillions of tokens, so they must learn general patterns (but can still repeat training text).
- **"The sky is…" failed:** ` sky` does not occur in my 10 documents, so the model has no number for it and cannot read the word.
- **100 completions of "The exchange's retail…":** 68 of 100 were the same sentence, copied from the text. The rest jumped into other documents mid-sentence, e.g. *"…represents 85% of the chocolate spread market. Kraft Foods…"*. The model knows which words usually follow each other, but not what the text is about. It sounds fluent but is put together wrongly, which is the basis of hallucinations.
- **Why different answers from the same model?** The next token is **drawn at random** from probabilities, with a new seed each time. That is why ChatGPT can answer the same question differently.
