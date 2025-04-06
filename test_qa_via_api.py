import pandas as pd
import requests
import time
import random
import json
import csv

# Config
QUESTIONS_CSV = 'varied_test_questions.csv'
OUTPUT_CSV = 'varied_qa_test_results.csv'
API_URL = 'http://localhost:8502/api/ask'  # Update with your public IP if needed

# Load test questions with more robust error handling
def load_questions():
    try:
        # Read raw lines first to handle any formatting issues
        with open(QUESTIONS_CSV, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Clean up lines and ensure proper CSV format
        cleaned_lines = ['question']  # header
        for line in lines[1:]:  # skip header
            line = line.strip()
            if line and '?' in line:
                # Quote the entire line to handle any commas
                cleaned_lines.append(f'"{line}"')
        
        # Write to temporary file
        temp_csv = 'temp_questions.csv'
        with open(temp_csv, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cleaned_lines))
        
        # Read the cleaned CSV
        questions_df = pd.read_csv(temp_csv)
        questions = questions_df['question'].dropna().unique().tolist()
        
        # Clean up
        import os
        os.remove(temp_csv)
        
        return questions
        
    except Exception as e:
        print(f"Error loading questions: {str(e)}")
        return []

# Function to call the FastAPI endpoint
def query_api(question):
    try:
        response = requests.post(API_URL, json={'query': question}, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        if "results" in data:
            return data["results"]
        else:
            return f"ERROR: Unexpected response format - {data}"
    except requests.exceptions.RequestException as e:
        return f"ERROR: API request failed - {str(e)}"
    except json.JSONDecodeError as e:
        return f"ERROR: Invalid JSON response - {str(e)}"
    except Exception as e:
        return f"ERROR: Unexpected error - {str(e)}"

# Run tests and collect responses
def run_tests(questions, num_samples=30):
    results = []
    if not questions:
        print("No valid questions found!")
        return results
        
    sample_size = min(num_samples, len(questions))
    for q in random.sample(questions, sample_size):
        print(f"Querying: {q}")
        answer = query_api(q)
        results.append({'question': q, 'answer': answer})
        time.sleep(1)  # throttle slightly
    return results

# Main execution
try:
    # Load questions
    questions = load_questions()
    print(f"Loaded {len(questions)} valid questions")
    
    # Run tests
    results = run_tests(questions)
    
    # Save results
    if results:
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)
        print(f"Saved results to {OUTPUT_CSV}")
    else:
        print("No results to save!")
    
except Exception as e:
    print(f"Error: {str(e)}")
