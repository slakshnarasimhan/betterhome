
import pandas as pd
import requests
import time
import random

# Config
QUESTIONS_CSV = 'generated_test_questions.csv'
OUTPUT_CSV = 'qa_test_results.csv'
API_URL = 'http://localhost:8502/api/ask'  # Update with your public IP if needed

# Load test questions
questions_df = pd.read_csv(QUESTIONS_CSV)
questions = questions_df['question'].dropna().unique().tolist()

# Function to call the FastAPI endpoint
def query_api(question):
    try:
        response = requests.post(API_URL, json={'query': question}, timeout=30)
        return response.json().get("results", "No answer returned")
    except Exception as e:
        return f"ERROR: {str(e)}"

# Run tests and collect responses
results = []
for q in random.sample(questions, min(30, len(questions))):  # Cap to 20 for test run
    print(f"Querying: {q}")
    answer = query_api(q)
    results.append({'question': q, 'answer': answer})
    time.sleep(1)  # throttle slightly

# Save results
pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"Saved results to {OUTPUT_CSV}")
