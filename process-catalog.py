import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from unidecode import unidecode
from bs4 import MarkupResemblesLocatorWarning
import warnings

# Suppress BeautifulSoup warning
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def clean_text(text):
    """
    Clean text by removing HTML tags, special characters, and normalizing unicode.
    Also, convert multi-line text into a single line.
    """
    if isinstance(text, str):
        text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
        text = unidecode(text)  # Normalize unicode characters
        text = re.sub(r'[\r\n\t]', ' ', text)  # Remove newlines, tabs, etc.
        text = re.sub(r'[^\w\s\-.,]', '', text)  # Remove special characters except ., - and ,
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text
    return text


def clean_data(file_path, output_file):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Display all columns read for verification
    print("Columns found in the CSV file:", df.columns.tolist())

    # Clean all text fields
    df = df.applymap(clean_text)

    # Generate URLs using handle
    if 'handle' in df.columns:
        df['url'] = 'https://betterhomeapp.com/products/' + df['handle']

    # Rename columns (Ensure 'Product Type' is captured properly)
    rename_mapping = {
        'brand_(product.metafields.custom.brand)': 'Brand',
        'product_type': 'Product Type',  # Changed from 'Type' to 'Product Type'
        'product_category': 'Category',
        'variant_sku': 'SKU',
        'variant_grams': 'Weight',
        'variant_price': 'Better Home Price',
        'variant_compare_at_price': 'Retail Price',
        'seo_description': 'Description',
        'product.metafields.custom.features': 'Features',
        'product.metafields.custom.material': 'Material',
        'returns_(product.metafields.custom.returns)': 'Returns Policy',
        'warranty_(product.metafields.custom.warranty)': 'Warranty',
        'product.metafields.custom.source': 'Manufacturer URL'
    }

    # Apply renaming directly without filtering
    df.rename(columns=rename_mapping, inplace=True)

    # Log renamed columns and check if 'Product Type' is present
    print("Columns after renaming:", df.columns.tolist())
    if 'Product Type' not in df.columns:
        print("Error: 'Product Type' was not renamed properly. Double-check column names.")
    else:
        print("Product Type field found and renamed successfully")
        print("Sample of Product Type values:", df['Product Type'].head())

    # Handle duplicate SKUs if SKU column exists
    if 'SKU' in df.columns:
        df.drop_duplicates(subset=['SKU'], keep='first', inplace=True)

    # Fill missing attributes for rows with the same handle
    if 'handle' in df.columns:
        for handle, group in df.groupby('handle'):
            for column in df.columns:
                if group[column].isnull().any():
                    filled_value = group[column].dropna().iloc[0] if not group[column].dropna().empty else np.nan
                    df.loc[df['handle'] == handle, column] = df.loc[df['handle'] == handle, column].fillna(filled_value)
            df = df.infer_objects(copy=False)  # Prevent FutureWarning for fillna()

    # Process Option Names for extracting attributes
    attributes = {'Color': [], 'Finish': [], 'Material': [], 'Style': []}
    option_columns = ['option1_name', 'option2_name', 'option3_name']
    option_value_columns = ['option1_value', 'option2_value', 'option3_value']

    for index, row in df.iterrows():
        for option_col, value_col in zip(option_columns, option_value_columns):
            option_name = row.get(option_col, '') if isinstance(row.get(option_col, ''), str) else ''
            option_value = row.get(value_col, '')

            if option_name.lower() in ['color', 'finish', 'material', 'style']:
                attributes[option_name.capitalize()].append(option_value)
                df.loc[index, option_name.capitalize()] = option_value

    # Keep only necessary columns
    columns_to_keep = [
        'handle', 'title', 'Product Type', 'Category', 'tags', 'SKU', 'Weight', 'Better Home Price',
        'Retail Price', 'Description', 'Brand', 'Features', 'Material', 'Returns Policy',
        'Manufacturer URL', 'Warranty', 'url', 'Color', 'Finish', 'Material', 'Style'
    ]

    existing_columns_in_df = [col for col in columns_to_keep if col in df.columns]
    cleaned_df = df[existing_columns_in_df]

    # Save the cleaned DataFrame to a new CSV file
    cleaned_df.to_csv(output_file, index=False)
    print(f"Data cleaned and saved to {output_file}")


if __name__ == "__main__":
    input_file = 'products_export_1 2.csv'  # Replace with your input file
    #input_file = 'test.csv'  # Replace with your input file
    output_file = 'cleaned_products.csv'  # Replace with your desired output file name
    clean_data(input_file, output_file)

