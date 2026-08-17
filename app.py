"""
app.py
------
Astrology RAG Assistant

Flow:
1. Ask user date of birth
2. Convert to query
3. Search FAISS vector database
4. Retrieve zodiac document
5. Send context to LLM
6. Generate explanation
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


# ─────────────────────────────────────────────
# Helper: Extract month
# ─────────────────────────────────────────────
def extract_month(date_input: str) -> str:
    months = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ]

    date_input = date_input.title()

    for m in months:
        if m in date_input:
            return m

    return date_input


# ─────────────────────────────────────────────
# Load FAISS Vector Database
# ─────────────────────────────────────────────
def load_vectorstore():

    if not os.path.exists("faiss_index"):
        raise Exception("Run build_index.py first!")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


# ─────────────────────────────────────────────
# Generate Answer using LLM
# ─────────────────────────────────────────────
def generate_answer(query, docs):

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a friendly astrology expert.

Using the zodiac information below, explain the person's zodiac sign.

Zodiac information:
{context}

User question:
{query}

Explain:
• zodiac sign name
• symbol
• date range
• personality traits

Make the answer friendly and conversational.
"""

    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0.7
    )

    response = llm.invoke(prompt)

    return response.content


# ─────────────────────────────────────────────
# Main Program
# ─────────────────────────────────────────────
def main():

    print("="*50)
    print("🔮 Astrology RAG Assistant")
    print("="*50)

    print("\nLoading vector database...")
    vectorstore = load_vectorstore()
    print("Ready!\n")

    while True:

        date_input = input(
            "Enter your date of birth (example: April 5) or 'quit': "
        ).strip()

        if date_input.lower() in ["quit","exit","q"]:
            print("Goodbye ✨")
            break

        if not date_input:
            continue

        month = extract_month(date_input)

        query = f"""
Person born on {date_input} in {month}.
What is their zodiac sign and personality traits?
"""

        print("\nSearching vector database...")

        docs = vectorstore.similarity_search(query, k=1)

        print("\nRetrieved zodiac profile:\n")
        print(docs[0].page_content)
        print("\nGenerating AI explanation...\n")

        answer = generate_answer(query, docs)

        print("─"*50)
        print(answer)
        print("─"*50)
        print()


if __name__ == "__main__":
    main()