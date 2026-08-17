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
