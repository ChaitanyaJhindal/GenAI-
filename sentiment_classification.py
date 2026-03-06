from openai import OpenAI
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)

text = "The phone battery life is terrible and I regret buying it."

prompt = f"""
You are a sentiment classifier.

Classify the sentiment of the given text into ONE of the following labels:
Positive
Negative
Neutral

Return ONLY the label.

Text: "{text}"
"""

response = client.responses.create(
    model="openai/gpt-oss-20b",
    input=prompt
)

print("Sentiment:", response.output_text)