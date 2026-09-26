# Celestia — Astrology RAG Assistant

Celestia is a retrieval-augmented astrology assistant built with Flask, LangChain, sentence-transformer embeddings, and FAISS. It detects zodiac signs from birth dates, retrieves grounded zodiac context, and generates conversational answers, compatibility insights, and horoscopes.

## Highlights

- Browser-based chat interface
- Local FAISS semantic search over curated zodiac data
- Confidence-aware retrieval and fallback handling
- Zodiac compatibility and horoscope endpoints
- CLI assistant and Flask web server

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
copy .env.example .env
python build_index.py
python server.py
```

Add your API key and a random Flask session secret to `.env`. Open `http://localhost:5000` after the server starts.

## Main files

- `server.py` — Flask API and web application
- `app.py` — command-line RAG assistant
- `build_index.py` — builds the FAISS index
- `zodiac_data.txt` — retrieval knowledge base

## Retrieval evaluation

`eval/eval_retrieval.py` measures retrieval quality offline, with no LLM or API key needed. It builds FAISS indexes with the same `all-MiniLM-L6-v2` embeddings, runs 60 test questions whose answers are known (12 signs × 5 questions: 3 name the sign, 2 are reverse lookups such as "Which sign is symbolised by the twins?"), and reports Hit@1, Hit@3, MRR@3 and field-level accuracy.

| Chunking (k = 3, cosine similarity) | Chunks | Hit@1 | Hit@3 | MRR@3 | Hit@1 reverse lookups |
|---|---|---|---|---|---|
| Whole sign (current index) | 12 | 70% | 80% | 0.74 | 25% |
| **Field-level (sign-prefixed)** | 180 | **90%** | **98%** | **0.94** | **79%** |
| Fixed 300 chars / 50 overlap | 92 | 73% | 90% | 0.82 | 38% |
| Fixed 800 chars / 100 overlap | 33 | 65% | 80% | 0.71 | 17% |

**Findings:** whole-sign chunks answer questions that name the sign perfectly but miss most reverse lookups, because a ~2,000-character chunk dilutes each individual fact in a single embedding. Field-level chunks, each prefixed with its sign, keep facts sharp and raise Hit@1 from 70% to 90%. MMR re-ranking gave no gain over plain similarity search on this data. Full results, including MMR, are in [`eval/RESULTS.md`](eval/RESULTS.md).

```
python eval/eval_retrieval.py
```
