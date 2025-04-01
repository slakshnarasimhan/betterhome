# regenerate_test_questions.py
import pandas as pd

df = pd.read_csv('cleaned_products.csv')  # Make sure this file is present

def generate_questions(row):
    questions = []
    if pd.notna(row['Product Type']):
        pt = row['Product Type']
        questions.append(f"What brands of {pt} do you have?")
        questions.append(f"What is the cheapest {pt}?")
        questions.append(f"What is the most expensive {pt}?")
        questions.append(f"Do you have {pt}?")
    if pd.notna(row['title']):
        questions.append(f"Tell me more about {row['title']}")
    return questions

unique_questions = set()
for _, row in df.iterrows():
    unique_questions.update(generate_questions(row))

questions_df = pd.DataFrame(list(unique_questions), columns=["question"])
questions_df.to_csv('generated_test_questions.csv', index=False)
print("generated_test_questions.csv created!")

