
import json
import os
import csv
import pathlib
import openai 
import pymupdf4llm
import rich
from pydantic import BaseModel

client = openai.OpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])
MODEL_NAME = os.getenv("GITHUB_MODEL", "openai/gpt-4o")


class Result(BaseModel):
    exam: str
    value: str
    ranges: str
    value_in_range: str


class Records(BaseModel):
    patient: str
    id: str
    age: str
    date: str
    result: list[Result]
    total_results: str
    total_not_in_range: str

#If the results were available in a CSV file, we could load them as follows:
with open("exam_results.csv") as file:
    reader = csv.reader(file)
    rows = list(reader)
documents = [{"id": (i + 1), "body": " ".join(row)} for i, row in enumerate(rows[1:])]

#If the results were in PDF files, we could extract them as follows:
data_dir = pathlib.Path(os.path.dirname(__file__)) / "data"
filenames = ["agosto_all.pdf"]
all_chunks = []
for filename in filenames:
    # Extract text from the PDF file
    md_text = pymupdf4llm.to_markdown(data_dir / filename)
    all_chunks.append(md_text)

completion = client.beta.chat.completions.parse(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "Extract the exam results. As the file it is in Spanish, remove the accents characters. "
        "Return a JSON object with the following fields: patient (string), id (string), age (string), date (string), result (a list of objects with exam (string), value (string), the range of the exam (string) and the value in range checking if the values satisfy the range condition ('Yes' or 'No')). "
        "Calculate total_results (string) as the total number of results."
        "Calculate total_not_in_range (string) as the number of results where value is not in range."
        "If any field is missing, return a refusal message indicating which field is missing."},
        {"role": "user", "content": f"Sources: {all_chunks}"},
    ],
    response_format=Records
)


message = completion.choices[0].message

if message.refusal:
    rich.print(message.refusal)
else:
    event = message.parsed
    rich.print(event)

    #Return a json file with the extracted data
    with open("extracted_results.json", "w") as f:
        json.dump(event.dict(), f, indent=4)    




    
