# Notes for the oral exam

Write the answers in your own words as you work through the steps. Each question is one the teacher gives as an exam example.

## Training data
*How could a model's training data affect its market analysis?*

- Træningsdata er almindelig tekst fra internettet (FineWeb). Mine 10 dokumenter var en tilfældig blanding: et forum om sæbeopera, medicinsk forskning, en restaurantinspektion, sportsnyheder, Kraft/Cadbury-chokolade. Der var også fejl i dem, f.eks. et ødelagt tegn (`�`).
- En model kan kun det, dens træningsdata indeholder. Min model kunne slet ikke læse "sky", fordi ordet ikke stod i teksterne (trin 3).
- **Erhverv:** træningsdata beskriver fortiden, frem til modellens "knowledge cutoff". En markedsanalyse fra modellens hukommelse kender ikke ændringer efter træningen (nye konkurrenter, priser, regler), og den afspejler det, der står meget om på internettet, ikke nødvendigvis det, der er rigtigt for netop vores marked. Aktuelle tal skal gives til modellen i konteksten.

## Training vs. inference
*Does uploading a company report change the model's weights?*

- **Træning** = vægtene ændres. I `train.py` blev modellens vægte justeret 1.484 gange på ca. 60 sekunder, hver gang den gættede forkert, og loss faldt fra 7,7 til 0,09. Resultatet blev gemt i `tiny_gpt.pt`.
- **Inferens** = modellen bruges, og vægtene ændres ikke. `sky_completions.py` og `corpus_completions.py` indlæste `tiny_gpt.pt` og genererede tekst, men ændrede ikke filen.
- **Erhverv:** nej. At uploade en årsrapport til ChatGPT er inferens: rapporten lægges i konteksten for netop dette svar, men modellen lærer den ikke permanent. I en ny samtale er den væk.

## Base models vs. assistants
*Why might a base model continue a customer's question instead of answering it?*

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

22,044 characters became 4,747 tokens which is 4.6 characters per token.

- **Tekst → bytes → bits:** computeren gemmer tekst som tal. `|Viewing Single Post From: Spoilers for the Week of February 11th|` er 66 tegn = 66 bytes (tal fra 0–255, f.eks. `|` = 124, `V` = 86) = 528 bits (0 og 1).
- **Tokens:** en LLM læser hverken bogstaver eller bytes, men tokens, dvs. stykker af tekst med et nummer. GPT-4o har ca. 200.000 forskellige tokens. Almindelige ord er ét token (` the`), sjældne ord deles i flere.
- **Chat-tokens:** 16 af mine 4.747 tokens var ikke min tekst, men Tiktokenizers chat-indpakning: `<|im_start|>system … You are a helpful assistant <|im_end|><|im_start|>user …` og til sidst `<|im_start|>assistant`. Sådan ser en assistent en samtale: specielle tokens markerer, hvem der taler, og til sidst står der "assistant", så modellen fortsætter som assistenten. Selve korpusset var 4.731 tokens.
- **Erhverv:** pris, hastighed og grænsen for kontekstvinduet måles i tokens. En lang rapport koster flere tokens og kan fylde konteksten op.

## What my own tiny model showed (step 3)
*Loss before/after, what the text looked like before and after training, why "The sky is…" may fail:*

- **Model:** en lille transformer (2 lag, 4 attention heads) med tilfældige vægte, trænet i ca. 60 sekunder på min CPU til at gætte næste token. Den brugte kun de ca. 2.000 forskellige tokens, der findes i mit korpus, ikke alle 200.000.
- **Loss** (hvor forkert den gætter): 7,71 → 4,35 (step 50) → 1,76 (step 100) → 0,39 (step 200) → **0,09** (step 1.484). 7,7 svarer til rent gætteri blandt ca. 2.000 tokens.
- **Før træning:** ordsalat, *"The focused received restrict SHARE nothing retail freshans vegetables…"*
- **Efter træning:** *"The UK product would be positioned as a snack. "The breakfast market in countries such as Germany is more developed…"*. Den sætning står **ord for ord** i Kraft/Cadbury-dokumentet. Modellen har lært teksterne **udenad** i stedet for at lære sprog, fordi den kun havde 4.731 tokens. Rigtige LLM'er træner på billioner af tokens, så de må lære generelle mønstre (men kan stadig gentage træningstekst).
- **"The sky is…" fejlede:** ` sky` findes ikke i mine 10 dokumenter, så modellen har intet nummer for det og kan ikke læse ordet.
- **100 fortsættelser af "The exchange's retail…":** 68 af 100 var den samme sætning, kopieret fra teksten. Resten sprang over i andre dokumenter midt i sætningen, f.eks. *"…represents 85% of the chocolate spread market. Kraft Foods…"*. Modellen ved, hvilke ord der plejer at følge efter hinanden, men ikke hvad teksten handler om. Det lyder flydende, men det er sat forkert sammen, og det er grundlaget for hallucinationer.
- **Hvorfor forskellige svar fra samme model?** Næste token **trækkes tilfældigt** ud fra sandsynligheder, med et nyt seed hver gang. Derfor kan ChatGPT svare forskelligt på samme spørgsmål.
