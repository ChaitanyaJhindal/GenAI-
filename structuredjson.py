from openai import OpenAI
import os
import time
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize client
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# 20 test inputs
texts = [
"I love this phone, it works perfectly.",
"The product stopped working after two days.",
"The delivery was okay, nothing special.",
"This laptop is extremely fast and reliable.",
"I regret buying this item.",
"The packaging was normal.",
"The camera quality is fantastic.",
"The service was terrible and slow.",
"The movie was average.",
"I am very happy with this purchase.",
"The battery dies quickly.",
"The experience was fine.",
"This is the best purchase I have made.",
"The software keeps crashing.",
"The quality is acceptable.",
"I absolutely hate the interface.",
"The design looks great.",
"The product is neither good nor bad.",
"This exceeded my expectations.",
"I would not recommend this product."
]

print("TEXT | SENTIMENT | LATENCY(sec) | PROMPT TOKENS | COMPLETION TOKENS | TOTAL TOKENS")
print("-"*100)

for text in texts:

    prompt = f"""
Classify the sentiment of the following text.

Return strictly in JSON format:

{{
  "sentiment": "Positive | Negative | Neutral"
}}

Text: "{text}"
"""

    start = time.time()

    response = client.responses.create(
        model="openai/gpt-oss-20b",
        input=prompt
    )

    latency = time.time() - start

    raw_output = response.output_text.strip()

    # Extract sentiment from JSON
    try:
        sentiment = json.loads(raw_output)["sentiment"]
    except:
        sentiment = raw_output

    prompt_tokens = None
    completion_tokens = None
    total_tokens = None

    try:
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens
        total_tokens = response.usage.total_tokens
    except:
        pass

    print(f"{text} | {sentiment} | {latency:.4f} | {prompt_tokens} | {completion_tokens} | {total_tokens}")