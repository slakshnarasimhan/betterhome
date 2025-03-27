import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from unidecode import unidecode


def clean_text(text):
    """
    Clean text by removing HTML tags, special characters, and normalizing unicode.
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

    # Clean all text fields
    df = df.applymap(clean_text)

    # Generate URLs using handle
    if 'handle' in df.columns:
        df['url'] = 'https://betterhomeapp.com/products/' + df['handle']

    # Rename columns (Corrected 'Brand' and 'Type' handling)
    df.rename(columns={
        'variant_sku': 'SKU',
        'variant_grams': 'Weight',
        'variant_price': 'Better Home Price',
        'variant_compare_at_price': 'Retail Price',
        'seo_description': 'Description',
        'brand_(product.metafields.custom.brand)': 'Brand',  # Correctly rename Brand
        'features_(product.metafields.custom.features)': 'Features',  # Correctly rename Brand
        'product_type': 'Type',  # Correctly rename Product Type
        'product.metafields.custom.material': 'Material',
        'product.metafields.custom.returns': 'Returns Policy',
        'product.metafields.custom.source': 'Manufacturer URL',
        'product.metafields.custom.warranty': 'Warranty'
    }, inplace=True)

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
        'handle', 'title', 'type', 'tags', 'SKU', 'Weight', 'Better Home Price',
        'Retail Price', 'Description', 'Brand', 'Color', 'Features', 'Material', 
        'Returns Policy', 'Manufacturer URL', 'Warranty', 'url', 'Color', 'Finish', 
        'Material', 'Style'
    ]

    existing_columns = [col for col in columns_to_keep if col in df.columns]
    cleaned_df = df[existing_columns]

    # Save the cleaned DataFrame to a new CSV file
    cleaned_df.to_csv(output_file, index=False)
    print(f"Data cleaned and saved to {output_file}")


if __name__ == "__main__":
    input_file = 'products_export_1 2.csv'  # Replace with your input file
    output_file = 'cleaned_products.csv'  # Replace with your desired output file name
    clean_data(input_file, output_file)

