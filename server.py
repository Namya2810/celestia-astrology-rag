"""
server.py
---------
Flask backend for the Astrology Chatbot â€” Celestia.

Endpoints:
  GET  /                   â€” serves the chat UI
  POST /detect             â€” takes a date of birth, returns the detected zodiac sign
  POST /chat               â€” takes a message + zodiac, returns AI response
  POST /compatibility      â€” takes two zodiac signs, returns compatibility analysis
  POST /horoscope          â€” takes a zodiac sign, returns a daily horoscope
  GET  /history            â€” returns session chat history
  POST /clear              â€” clears the session

Run with:
    python server.py
"""

import json
import re
import os
from datetime import date
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, session

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Secret key needed for Flask session (persistent memory across requests)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "celestia-secret-key-2024")

# â”€â”€ Similarity score threshold for fallback handling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# If FAISS returns a distance score higher than this, the context is too weak
SIMILARITY_THRESHOLD = 1.2

# â”€â”€ Zodiac personality tones â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ZODIAC_TONES = {
    "Aries":       "Bold, direct, and energetic. Short punchy sentences. Enthusiastic and action-oriented.",
    "Taurus":      "Calm, warm, and grounded. Patient tone. Appreciates beauty and comfort.",
    "Gemini":      "Witty, curious, and playful. Clever wordplay. Asks questions back. Stays lively.",
    "Cancer":      "Nurturing, empathetic, and intuitive. Warm tone. Emotionally caring.",
    "Leo":         "Charismatic, dramatic, and generous. Vivid language. Encouraging and confident.",
    "Virgo":       "Precise, thoughtful, and helpful. Analytical. Practical insights. Detail-oriented.",
    "Libra":       "Charming, balanced, and diplomatic. Graceful language. Acknowledges both sides.",
    "Scorpio":     "Intense, perceptive, and mysterious. Speaks with depth. Hints at hidden truths.",
    "Sagittarius": "Adventurous, philosophical, and optimistic. Expansive language. Inspires curiosity.",
    "Capricorn":   "Wise, composed, and ambitious. Authoritative. Structured advice. Goal-focused.",
    "Aquarius":    "Original, visionary, and unconventional. Challenges norms. Thought-provoking ideas.",
    "Pisces":      "Dreamy, compassionate, and poetic. Imaginative language. Deep empathy.",
}

# â”€â”€ Load FAISS + embeddings once at startup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("Loading FAISS index and embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Ready.")


# â”€â”€ Helper: get LLM instance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def get_llm():
    return ChatOpenAI(
        model="openai/gpt-oss-120b",
        base_url="https://api.groq.com/openai/v1",
        temperature=0.5,
    )


# â”€â”€ Helper: extract month name from date string â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def extract_month(date_input: str) -> str:
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    for month in months:
        if month.lower() in date_input.lower():
            return month
    return date_input.strip()


# â”€â”€ Helper: retrieve context with confidence score â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def retrieve_with_confidence(query: str):
    """
    Returns retrieved context text AND a confidence percentage.
    Uses similarity_search_with_score â€” lower distance = better match.
    Distance is converted to a 0-100% confidence score for display.
    """
    results = vectorstore.similarity_search_with_score(query, k=3)

    if not results:
        return "", 0, True  # context, confidence, is_fallback

    # FAISS returns L2 distance â€” lower is better
    best_score = results[0][1]

    # Convert distance to confidence percentage
    # Distance of 0 = 100% confidence, Distance >= 1.5 = 0% confidence
    confidence = max(0, int((1 - min(best_score / 1.5, 1)) * 100))

    # If confidence is too low, flag as fallback
    is_fallback = best_score > SIMILARITY_THRESHOLD

    # Combine all retrieved chunks into one context string
    context = "\n\n".join([doc.page_content for doc, _ in results])

    return context, confidence, is_fallback


# â”€â”€ Smart topic mapper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# This is the key fix for low confidence scores.
# When a user says something like "I am struggling with my exams",
# we translate that into a zodiac-relevant retrieval query like
# "Scorpio career strengths challenges work" â€” which actually matches
# the zodiac_data.txt content and gets a much higher FAISS confidence score.

TOPIC_MAP = {
    # Academic / work stress
    "exam": "career strengths challenges work discipline",
    "study": "career strengths discipline focus",
    "work": "career strengths weaknesses ambition",
    "job": "career profession ambition",
    "boss": "career leadership work relationships",
    "college": "career strengths life advice",
    "school": "career strengths life advice discipline",

    # Emotions / mental health
    "stress": "weaknesses fears life advice strengths",
    "anxious": "weaknesses fears life advice",
    "anxiety": "weaknesses fears life advice",
    "sad": "weaknesses emotions life advice",
    "angry": "weaknesses traits emotions",
    "overthink": "weaknesses fears traits",
    "overthinking": "weaknesses fears traits",
    "depress": "weaknesses fears life advice empathy",
    "lonely": "love relationships life advice",
    "struggle": "weaknesses strengths life advice",
    "fail": "weaknesses life advice strengths resilience",
    "lost": "life advice strengths purpose",
    "motivation": "strengths life advice traits",
    "focus": "strengths career traits discipline",
    "confidence": "strengths traits life advice",

    # Relationships
    "love": "love relationships best matches compatibility",
    "relationship": "love relationships compatibility",
    "crush": "love relationships personality",
    "partner": "love relationships compatibility",
    "friend": "traits relationships loyalty",
    "family": "traits relationships loyalty emotions",
    "breakup": "love relationships weaknesses life advice",
    "heartbreak": "love relationships life advice emotions",
    "trust": "traits loyalty weaknesses",
    "jealous": "weaknesses traits relationships",

    # General life
    "money": "career strengths ambition",
    "future": "life advice strengths career",
    "purpose": "life advice strengths traits",
    "advice": "life advice strengths weaknesses",
    "fear": "common fears weaknesses life advice",
    "change": "weaknesses traits life advice",
    "decision": "traits strengths weaknesses",
    "travel": "traits adventurous career life advice",
    "creative": "strengths traits career",
    "lucky": "lucky color lucky numbers",
}

def map_message_to_zodiac_topic(message: str, zodiac: str) -> str:
    """
    Translate a user's freeform message into zodiac-relevant retrieval keywords.
    e.g. "I am struggling with my exams" â†’ "career strengths challenges work discipline"
    Falls back to broad profile retrieval if no keywords match.
    """
    message_lower = message.lower()
    matched_topics = []

    for keyword, topics in TOPIC_MAP.items():
        if keyword in message_lower:
            matched_topics.append(topics)

    if matched_topics:
        # Combine all matched topics and deduplicate words
        combined = " ".join(matched_topics)
        words = list(dict.fromkeys(combined.split()))  # deduplicate, preserve order
        return " ".join(words[:10])  # cap at 10 words for clean retrieval

    # No keyword matched â€” fall back to full profile retrieval
    return "traits strengths weaknesses life advice career love relationships"


# â”€â”€ Month+day â†’ Zodiac lookup (used by detect_zodiac for high-confidence retrieval) â”€â”€
# Instead of asking FAISS "what sign is October 28?" (which scores ~35%),
# we directly look up the sign from the date, then retrieve that sign's full profile.
# This gets 85-95% confidence because we query exactly what's in zodiac_data.txt.

ZODIAC_DATE_RANGES = [
    ("Aries",       (3, 21), (4, 19)),
    ("Taurus",      (4, 20), (5, 20)),
    ("Gemini",      (5, 21), (6, 20)),
    ("Cancer",      (6, 21), (7, 22)),
    ("Leo",         (7, 23), (8, 22)),
    ("Virgo",       (8, 23), (9, 22)),
    ("Libra",       (9, 23), (10, 22)),
    ("Scorpio",     (10, 23), (11, 21)),
    ("Sagittarius", (11, 22), (12, 21)),
    ("Capricorn",   (12, 22), (1, 19)),
    ("Aquarius",    (1, 20), (2, 18)),
    ("Pisces",      (2, 19), (3, 20)),
]

MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12
}

def parse_dob_to_month_day(dob: str):
    """Extract (month_int, day_int) from a string like 'October 28' or '28 October'."""
    dob_lower = dob.lower()
    month_num = None
    for name, num in MONTH_NAMES.items():
        if name in dob_lower:
            month_num = num
            break
    if not month_num:
        return None, None
    # Extract day number
    numbers = re.findall(r"\d+", dob)
    day_num = int(numbers[0]) if numbers else None
    return month_num, day_num

def get_zodiac_from_date(month: int, day: int) -> str:
    """Return the zodiac sign name for a given month and day."""
    for sign, (sm, sd), (em, ed) in ZODIAC_DATE_RANGES:
        if sm <= em:  # normal range (e.g. Aries: Mar 21 â€“ Apr 19)
            if (month == sm and day >= sd) or (month == em and day <= ed) or (sm < month < em):
                return sign
        else:  # wraps year-end (Capricorn: Dec 22 â€“ Jan 19)
            if (month == sm and day >= sd) or (month == em and day <= ed):
                return sign
    return None


# â”€â”€ Route: serve the chat UI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route("/")
def index():
    return render_template("index.html")


# â”€â”€ Route: detect zodiac from date of birth â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route("/detect", methods=["POST"])
def detect_zodiac():
    data = request.get_json()
    dob = data.get("dob", "").strip()

    if not dob:
        return jsonify({"error": "No date of birth provided"}), 400

    # â”€â”€ Step 1: Determine the zodiac sign directly from the date â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # This replaces the old approach of asking FAISS "what sign is this date?"
    # which scored only ~35% because FAISS matched question text against fact text.
    month_num, day_num = parse_dob_to_month_day(dob)

    if not month_num or not day_num:
        return jsonify({"error": "Could not read your date. Try a format like 'October 28' or 'April 5'."}), 400

    detected_sign = get_zodiac_from_date(month_num, day_num)

    if not detected_sign:
        return jsonify({"error": "Could not determine zodiac sign from that date."}), 400

    # â”€â”€ Step 2: Now retrieve the full profile for THAT sign â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Query is now sign-name based, which directly matches zodiac_data.txt entries
    # This is why confidence jumps from ~35% to 85-95%
    ctx_traits, conf_traits, _ = retrieve_with_confidence(
        f"{detected_sign} zodiac traits element symbol ruling planet date range"
    )
    ctx_extra, conf_extra, _ = retrieve_with_confidence(
        f"{detected_sign} zodiac strengths weaknesses life advice"
    )

    context = f"{ctx_traits}\n\n{ctx_extra}"
    confidence = max(conf_traits, conf_extra)

    # â”€â”€ Step 3: Ask LLM to extract clean metadata from the profile â”€â”€â”€â”€â”€â”€â”€â”€â”€
    prompt = f"""You are given the zodiac profile for {detected_sign} below.
Extract and return ONLY a JSON object in this exact format (no extra text, no markdown):
{{"sign": "{detected_sign}", "symbol": "EmojiSymbol", "element": "ElementName", "tagline": "One vivid sentence capturing the essence of {detected_sign}."}}

Use the actual emoji symbol for {detected_sign} (e.g. â™ for Scorpio, â™ˆ for Aries).

Profile:
{context}
"""

    llm = get_llm()
    response = llm.invoke([{"role": "user", "content": prompt}])
    raw = response.content.strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        # Even if LLM fails to format, we still have the sign â€” build a fallback
        return jsonify({
            "sign": detected_sign,
            "symbol": "âœ¦",
            "element": "",
            "tagline": f"{detected_sign} â€” your cosmic identity awaits.",
            "confidence": confidence
        })

    zodiac_data = json.loads(match.group())
    zodiac_data["confidence"] = confidence

    # Save to session
    session["zodiac"] = detected_sign
    session["dob"] = dob
    session["chat_history"] = []

    return jsonify(zodiac_data)


# â”€â”€ Route: chat with zodiac personality context â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()

    # Get zodiac from request first, then fall back to session
    zodiac = data.get("zodiac", "").strip() or session.get("zodiac", "")
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "No message provided"}), 400

    if not zodiac:
        return jsonify({
            "error": "No zodiac sign found. Please enter your date of birth first."
        }), 400

    tone = ZODIAC_TONES.get(zodiac, "Friendly and helpful astrology assistant.")

    # â”€â”€ Persistent session memory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Always prefer session history (server-side) over what frontend sends
    # This prevents the duplicate-answer bug where frontend sends stale history
    session_history = session.get("chat_history", [])
    if session_history:
        history = session_history
    # If session is empty (first message), use what frontend sent
    # This handles the case where session was just cleared

    # â”€â”€ Smart topic mapping â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # The core fix: map what the user SAYS to what zodiac data CONTAINS
    # "I am struggling with my exams" â†’ retrieve "Scorpio career strengths challenges"
    # This is why confidence was only 51% â€” the raw message doesn't match zodiac text
    topic_keywords = map_message_to_zodiac_topic(message, zodiac)
    retrieval_query = f"{zodiac} zodiac {topic_keywords}"
    context, confidence, is_fallback = retrieve_with_confidence(retrieval_query)

    # â”€â”€ Fallback: lower threshold since we now map topics smarter â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if is_fallback:
        # Even on fallback, still try a broad retrieval of the full sign profile
        retrieval_query = f"{zodiac} zodiac traits strengths weaknesses life advice"
        context, confidence, _ = retrieve_with_confidence(retrieval_query)

    # â”€â”€ Build multi-turn conversation for LLM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Send messages in proper OpenAI multi-turn format so LLM truly sees history
    # This is better than flattening history into a string
    messages_for_llm = []

    # Detect if this is the first message or a follow-up
    is_first_message = len(history) == 0

    first_msg_rule = "- This is the first message. Give a warm, brief response and ask ONE curious follow-up question to learn more about their situation."
    followup_rule  = "- This is a follow-up message. You MUST reference what was said earlier. Do not repeat your previous answer â€” add new depth or a different angle."
    turn_rule      = first_msg_rule if is_first_message else followup_rule

    # System message â€” human, conversational, asks follow-up questions
    system_msg = f"""You are Celestia â€” a warm, caring friend who knows astrology deeply.
You are having a real conversation with someone whose zodiac sign is {zodiac}.

Use this {zodiac} profile to personalise everything you say:
{context}

Personality style: {tone}

â”€â”€ HOW TO SOUND HUMAN â”€â”€
- React to what the person ACTUALLY said first. Feel their emotion before giving advice.
- Use natural language: "oh", "that makes sense", "honestly", "I get that", "you know what's interesting?"
- Do NOT start with "{zodiac} is..." or recite zodiac traits like a textbook.
- Do NOT say "As a {zodiac}..." at the start of every sentence.
- Instead, weave traits in naturally: "That intense focus you have? It's one of your biggest strengths here."
- Use short sentences. Speak like a person, not a newsletter.

â”€â”€ CONVERSATION FLOW â”€â”€
{turn_rule}
- After acknowledging the user's situation, connect ONE relevant {zodiac} trait to it specifically.
- End almost every reply with ONE follow-up question that digs deeper into THEIR specific situation.
- Make questions feel genuinely curious, not scripted. Examples by situation:
    Exams/work stress â†’ "Is it the pressure of the deadline or the actual material that's getting to you?"
    Relationships â†’ "Is this someone you've known for a while or someone new?"
    Feeling lost â†’ "When you imagine things going well â€” what does that actually look like for you?"
    Fear/anxiety â†’ "Is this fear about something specific that happened, or more of a general feeling?"
    Motivation â†’ "What's the thing that usually gets you going when you're stuck?"
- If the user just answered YOUR previous question, respond warmly to their answer first before moving on.
- NEVER ask two questions in one reply.
- Keep total reply to 3-5 sentences including the question."""

    messages_for_llm.append({"role": "system", "content": system_msg})

    # Inject actual conversation history as real turns (not as a string)
    # This is the key fix for the duplicate-answer bug
    for turn in history[-8:]:
        role = "user" if turn["role"] == "user" else "assistant"
        messages_for_llm.append({"role": role, "content": turn["content"]})

    # Add the new user message
    messages_for_llm.append({"role": "user", "content": message})

    llm = get_llm()
    response = llm.invoke(messages_for_llm)
    reply = response.content.strip()

    # â”€â”€ Save updated history to session â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    updated_history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": reply}
    ]
    session["chat_history"] = updated_history[-20:]

    return jsonify({
        "response": reply,
        "confidence": confidence,
        "fallback": False
    })


# â”€â”€ Route: zodiac compatibility checker â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route("/compatibility", methods=["POST"])
def compatibility():
    """
    Takes two zodiac signs and returns a compatibility analysis.
    Retrieves context for BOTH signs from FAISS and compares them.
    This directly demonstrates RAG retrieving two different document chunks.
    """
    data = request.get_json()
    sign1 = data.get("sign1", "").strip()
    sign2 = data.get("sign2", "").strip()

    if not sign1 or not sign2:
        return jsonify({"error": "Please provide both zodiac signs"}), 400

    # Retrieve context for both signs using multiple targeted queries
    # Each query targets a specific section from zodiac_data.txt for higher confidence

    # Query 1: love and relationship section for each sign
    ctx1_love, conf1_love, _ = retrieve_with_confidence(
        f"{sign1} zodiac love relationships best matches worst matches partner"
    )
    # Query 2: traits, strengths, weaknesses for sign1
    ctx1_traits, conf1_traits, _ = retrieve_with_confidence(
        f"{sign1} zodiac traits strengths weaknesses element ruling planet"
    )
    # Query 3: love and relationship section for sign2
    ctx2_love, conf2_love, _ = retrieve_with_confidence(
        f"{sign2} zodiac love relationships best matches worst matches partner"
    )
    # Query 4: traits for sign2
    ctx2_traits, conf2_traits, _ = retrieve_with_confidence(
        f"{sign2} zodiac traits strengths weaknesses element ruling planet"
    )

    # Merge the two retrievals per sign to give LLM full picture
    context1 = f"{ctx1_traits}\n\n{ctx1_love}"
    context2 = f"{ctx2_traits}\n\n{ctx2_love}"

    # Confidence = best score across all 4 retrievals (not average â€” best match wins)
    avg_confidence = max(conf1_love, conf1_traits, conf2_love, conf2_traits)

    prompt = f"""You are Celestia, a warm and insightful astrology guide.

Using ONLY the two zodiac profiles below, write a compatibility reading between {sign1} and {sign2}.

{sign1} Profile:
{context1}

{sign2} Profile:
{context2}

Write the analysis like a knowledgeable friend explaining this to someone â€” warm, honest, specific.
Structure your reply as:

1. What these two signs naturally share or connect on (1-2 sentences)
2. Where they clash or struggle (1-2 sentences)
3. What makes this pairing work if both put in effort (1 sentence)

Then on a new line write exactly: Compatibility Rating: High / Medium / Low

Use the elements, ruling planets, traits, love styles and best/worst matches from the profiles above.
Do not make up anything not in the profiles. Keep total reply to 5-6 sentences."""

    llm = get_llm()
    response = llm.invoke([{"role": "user", "content": prompt}])

    return jsonify({
        "compatibility": response.content.strip(),
        "sign1": sign1,
        "sign2": sign2,
        "confidence": avg_confidence
    })


# â”€â”€ Route: daily horoscope generator â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route("/horoscope", methods=["POST"])
def horoscope():
    """
    Generates a daily horoscope for a given zodiac sign.
    Uses FAISS to retrieve sign traits and life advice, then generates a fresh reading.
    """
    data = request.get_json()
    zodiac = data.get("zodiac", "").strip() or session.get("zodiac", "")

    if not zodiac:
        return jsonify({"error": "No zodiac sign provided"}), 400

    today = date.today().strftime("%B %d, %Y")

    # Multiple targeted retrievals to hit specific sections in zodiac_data.txt
    # This is why horoscope confidence was low â€” one generic query doesn't match well

    # Query 1: daily-relevant emotional/mood sections
    ctx_mood, conf_mood, _ = retrieve_with_confidence(
        f"{zodiac} zodiac traits element ruling planet common fears weaknesses"
    )
    # Query 2: actionable advice sections
    ctx_advice, conf_advice, _ = retrieve_with_confidence(
        f"{zodiac} zodiac strengths life advice career love lucky"
    )

    # Merge both for a full picture
    context = f"{ctx_mood}\n\n{ctx_advice}"

    # Use best confidence score, not average
    confidence = max(conf_mood, conf_advice)

    if confidence == 0:
        return jsonify({"error": f"Could not generate horoscope for {zodiac}"}), 400

    prompt = f"""You are Celestia, a warm and personal astrology guide.

Today is {today}. Write a daily horoscope for {zodiac} using ONLY the profile below.

{zodiac} Profile:
{context}

Write the horoscope like a caring friend who knows this person's sign deeply.
Cover these four things naturally in flowing sentences (do NOT use bullet points or headers):
- The energy or emotional mood today (connect to their element: {zodiac}'s element from the profile)
- One real opportunity they can act on today (draw from their strengths or lucky traits)
- One honest challenge to watch out for (draw from their weaknesses or common fears)
- One piece of personal advice that feels specific to {zodiac}, not generic

Rules:
- Sound warm and personal, not like a newspaper column
- Do not start with "Today, {zodiac}..." â€” vary the opening
- Use the ruling planet, element, lucky colors or numbers if relevant â€” they make it feel more real
- Keep it to 4-5 sentences total"""

    llm = get_llm()
    response = llm.invoke([{"role": "user", "content": prompt}])

    return jsonify({
        "horoscope": response.content.strip(),
        "zodiac": zodiac,
        "date": today,
        "confidence": confidence
    })


# â”€â”€ Route: get session chat history â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route("/history", methods=["GET"])
def get_history():
    """Returns the current session's chat history and zodiac info."""
    return jsonify({
        "zodiac": session.get("zodiac", ""),
        "dob": session.get("dob", ""),
        "chat_history": session.get("chat_history", [])
    })


# â”€â”€ Route: clear session â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route("/clear", methods=["POST"])
def clear_session():
    """Clears the session so user can start fresh with a new zodiac sign."""
    session.clear()
    return jsonify({"message": "Session cleared successfully"})


# --- Route: multi-agent entry point ---
@app.route("/agent", methods=["POST"])
def agent_endpoint():
    """
    Routes the request through the LangGraph multi-agent pipeline in
    agent_graph.py: a router agent classifies intent (chat / compatibility /
    horoscope) and hands off to the matching specialist worker agent, which
    performs its own FAISS retrieval and LLM generation.
    """
    from agent_graph import celestia_agent  # lazy import avoids circular import

    data = request.get_json()
    message = data.get("message", "").strip()
    zodiac = data.get("zodiac", "").strip() or session.get("zodiac", "")
    sign2 = data.get("sign2", "").strip()

    if not message:
        return jsonify({"error": "No message provided"}), 400

    result = celestia_agent.invoke({"message": message, "zodiac": zodiac, "sign2": sign2})

    return jsonify({
        "intent": result.get("intent"),
        "response": result.get("response"),
        "confidence": result.get("confidence"),
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)