import pandas as pd

VALIDATED_CSV = 'qa_test_results_validated.csv'
SUGGESTIONS_FILE = 'diagnostic_suggestions.md'

# Read the CSV file
df = pd.read_csv(VALIDATED_CSV)

# Generate summary
summary = df['issue'].value_counts().to_dict()
problem_questions = df[df['issue'] != 'Looks OK']

suggestions = []

suggestions.append("# Diagnostic Suggestions based on QA Output\n")

for issue, count in summary.items():
    suggestions.append(f"## Issue: {issue} ({count} occurrences)\n")
    
    sample_qs = problem_questions[problem_questions['issue'] == issue]['question'].head(3).tolist()
    suggestions.append("**Sample Questions:**")
    for q in sample_qs:
        suggestions.append(f"- {q}")
    
    if issue == "Empty or Error":
        suggestions.append("**Cause:** Query may not be reaching the backend or the backend is not returning a response.")
        suggestions.append("**Fix:** Check Streamlit query routing. Make sure the query input is captured and processed.")
    
    elif issue == "Product Not Found":
        suggestions.append("**Cause:** Likely a product type mismatch or incomplete synonym handling.")
        suggestions.append("**Fix:** Update `find_product_type()` in `ask-questions-updated-corrected.py` to add synonyms for these types.")
    
    elif issue == "No Price Info":
        suggestions.append("**Cause:** Missing price fields or formatting issues in response.")
        suggestions.append("**Fix:** In `format_product_response()` or the OpenAI system prompt, enforce price formatting with BetterHome and Retail prices.")
    
    elif issue == "No Brand Info":
        suggestions.append("**Cause:** Incomplete embedding context or lack of brand emphasis.")
        suggestions.append("**Fix:** Update `prepare_entries()` in `generate-embedding.py` to always include brand explicitly. Also refine the system prompt in `retrieve_and_generate_openai()`.")
    
    suggestions.append("\n---\n")

# Save suggestions to a markdown file
with open(SUGGESTIONS_FILE, 'w') as f:
    f.write("\n".join(suggestions))

print(f"Diagnostic suggestions written to {SUGGESTIONS_FILE}")
