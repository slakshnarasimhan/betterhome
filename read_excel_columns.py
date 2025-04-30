import pandas as pd

# Read the Excel file
file_path = 'betterhome-order-form.xlsx'
df = pd.read_excel(file_path)

# Print the column names
print(df.columns.tolist()) 