from openai import OpenAI
import os
import time
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

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

print("TEXT | SENTIMENT | LATENCY | PROMPT TOKENS | COMPLETION TOKENS | TOTAL TOKENS")
print("-"*95)

for text in texts:

    prompt =  f"""
You are a sentiment classification system.

Examples:

Text: "This product is amazing and works flawlessly."
Sentiment: Positive

Text: "The item broke after one day."
Sentiment: Negative

Text: "The experience was average."
Sentiment: Neutral

Now classify the following text.

Text: "{text}"
Sentiment:
"""

    start = time.time()

    response = client.responses.create(
        model="openai/gpt-oss-20b",
        input=prompt
    )

    latency = time.time() - start

    sentiment = response.output_text.strip()

    print(f"{text} | {sentiment} | {latency:.4f} | {response.usage.input_tokens} | {response.usage.output_tokens} | {response.usage.total_tokens}")