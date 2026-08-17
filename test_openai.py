"""
test_openai.py
--------------
Quickly verifies that your OpenAI API key is loaded correctly
and that you can reach the OpenAI API.

Run with:
    python test_openai.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

print("Testing OpenAI connection...")
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0
)
response = llm.invoke("Say 'OpenAI connection successful!' and nothing else.")
print(response.content)
