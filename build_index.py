"""
build_index.py
--------------
Reads zodiac_data.txt, splits it into one chunk per zodiac sign,
generates OpenAI embeddings, and saves a FAISS vector index to disk.

Run this ONCE before using app.py:
    python build_index.py
"""

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()  # Load OPENAI_API_KEY from .env

# ── 1. Load the raw text file ──────────────────────────────────────────────
print("Loading zodiac data...")
loader = TextLoader("zodiac_data.txt", encoding="utf-8")
documents = loader.load()

# ── 2. Split into one chunk per zodiac sign ────────────────────────────────
# Each sign in zodiac_data.txt is separated by "---", so we split on that.
splitter = CharacterTextSplitter(
    separator="---",
    chunk_size=500,      # large enough to hold one full sign description
    chunk_overlap=0,     # no overlap needed — each sign is self-contained
)
chunks = splitter.split_documents(documents)

print(f"  Created {len(chunks)} chunks (expected 12, one per zodiac sign)")

# ── 3. Create embeddings ───────────────────────────────────────────────────
print("Generating embeddings with OpenAI...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# ── 4. Build the FAISS index and save it to disk ───────────────────────────
print("Building FAISS index...")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")

print("Done! Index saved to ./faiss_index/")
print("You can now run app.py")
