from openai import OpenAI
import os
import csv
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

TOTAL_REAL = 75
TOTAL_FAKE = 75

dataset = []

def generate_headline(label):

    prompt = f"""
Generate ONE news headline.

Label: {label}

Rules:
- Topics: politics, science, technology, health, business, sports
- Headline must look like real news
- If label=fake, include unrealistic or misleading claims
- If label=real, it must sound believable

Return STRICTLY in this format:
headline,label

Example outputs:
NASA launches new climate monitoring satellite,real
Scientists claim plant discovered that cures aging overnight,fake
"""

    response = client.responses.create(
        model="openai/gpt-oss-20b",
        input=prompt
    )

    text_output = ""

    for block in response.output:
        if block.type == "message":
            for part in block.content:
                if part.type == "output_text":
                    text_output += part.text

    return text_output.strip()


print("Generating REAL headlines...")

for _ in range(TOTAL_REAL):
    row = generate_headline("real")

    if "," in row:
        headline, label = row.rsplit(",", 1)

        if label.strip() == "real":
            dataset.append([headline.strip(), label.strip()])

print("Generating FAKE headlines...")

for _ in range(TOTAL_FAKE):
    row = generate_headline("fake")

    if "," in row:
        headline, label = row.rsplit(",", 1)

        if label.strip() == "fake":
            dataset.append([headline.strip(), label.strip()])


with open("fake_news_dataset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["headline", "label"])
    writer.writerows(dataset)

print("Dataset generated successfully")
print("Total samples:", len(dataset))
print("Saved as fake_news_dataset.csv")