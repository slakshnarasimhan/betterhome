import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from unidecode import unidecode
from bs4 import MarkupResemblesLocatorWarning
import warnings
from typing import Dict

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


def remove_non_ascii(text):
    if isinstance(text, str):
        return re.sub(r'[^\x00-\x7F]+', '', text)
    return text


def parse_features_to_columns(features_str: str) -> str:
    """Convert features string into a single string with features as key:value pairs separated by '|', and remove non-ASCII characters."""
    if not isinstance(features_str, str):
        return ""
    features_list = [line.strip() for line in features_str.split('\n') if line.strip()]
    features_joined = '|'.join(features_list)
    return remove_non_ascii(features_joined)


def clean_data(file_path, output_file):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Display all columns read for verification
    print("Columns found in the CSV file:", df.columns.tolist())

    # Clean all text fields except image URLs and features (both original and renamed)
    feature_cols = ['features_(product.metafields.custom.features)', 'features']
    for column in df.columns:
        if column not in ['image_src'] + feature_cols:  # Skip cleaning image URLs and features
            df[column] = df[column].apply(clean_text)

    # Generate URLs using handle
    if 'handle' in df.columns:
        df['url'] = 'https://betterhomeapp.com/products/' + df['handle']

    # Parse and expand features into separate columns
    features_column = 'features_(product.metafields.custom.features)'
    if features_column in df.columns:
        print("\n[DEBUG] Processing features column:", features_column)
        
        # Parse features directly into 'Features' column
        df['Features'] = df[features_column].apply(parse_features_to_columns)
        
        print("\n[DEBUG] Features by handle before propagation:")
        for handle in df['handle'].unique():
            features = df[df['handle'] == handle]['Features']
            non_empty = features[features.notna() & (features.str.strip() != '')]
            print(f"\nHandle: {handle}")
            print("Number of rows:", len(features))
            print("Non-empty features found:", len(non_empty))
            if not non_empty.empty:
                print("First non-empty value:", non_empty.iloc[0])
        
        # Propagate non-empty Features to all rows with same handle
        print("\n[DEBUG] Propagating Features:")
        for handle in df['handle'].unique():
            mask = df['handle'] == handle
            features = df.loc[mask, 'Features']
            non_empty = features[features.notna() & (features.str.strip() != '')]
            if not non_empty.empty:
                value_to_propagate = non_empty.iloc[0]
                print(f"\nHandle: {handle}")
                print(f"Propagating value: {value_to_propagate}")
                df.loc[mask, 'Features'] = value_to_propagate

    # Debug: Print Features for a specific handle after parsing but before propagation
    debug_handle = 'atomberg-renesa-halo-1200-mm-bldc-ceiling-fan-with-remote-control-led-indicators'
    if 'handle' in df.columns and ('features' in df.columns or 'Features' in df.columns):
        feature_col = 'features' if 'features' in df.columns else 'Features'
        print(f"\n[DEBUG] Features for handle '{debug_handle}' after parsing but before propagation:")
        print(df.loc[df['handle'] == debug_handle, feature_col])

    # Debug: Print all DataFrame columns after loading
    print("[DEBUG] Columns after loading:", df.columns.tolist())

    # Rename columns
    rename_mapping = {
        'brand_(product.metafields.custom.brand)': 'Brand',
        'type': 'Product Type',
        'product_category': 'Category',
        'variant_sku': 'SKU',
        'variant_grams': 'Weight',
        'variant_price': 'Better Home Price',
        'variant_compare_at_price': 'Retail Price',
        'seo_description': 'Description',
        'material_(product.metafields.custom.material)': 'Material',
        'returns_(product.metafields.custom.returns)': 'Returns Policy',
        'warranty_(product.metafields.custom.warranty)': 'Warranty',
        'source_(product.metafields.custom.source)': 'Manufacturer URL',
        'image_src': 'Image Src',
        'image_alt_text': 'Image Alt Text'
    }

    # Apply renaming directly without filtering
    df.rename(columns=rename_mapping, inplace=True)

    # After renaming columns
    print("[DEBUG] Columns after renaming:", df.columns.tolist())

    # Debug: Print which column is being used for features
    feature_col = None
    for col in df.columns:
        if 'features' in col.lower():
            feature_col = col
            break
    print(f"[DEBUG] Feature column detected for parsing: {feature_col}")

    # Debug: Print all unique handles
    if 'handle' in df.columns:
        print("[DEBUG] Unique handles in DataFrame:", df['handle'].unique())
    # Debug: For each handle, print features values before propagation
    if 'handle' in df.columns and feature_col:
        for handle in df['handle'].unique():
            features_for_handle = df.loc[df['handle'] == handle, feature_col]
            print(f"[DEBUG] Features for handle '{handle}' before propagation:")
            print(features_for_handle.values)

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
            df = df.infer_objects(copy=False)

    # Use groupby().transform() to robustly propagate the first non-empty Features value to all rows with the same handle
    if 'handle' in df.columns and 'Features' in df.columns:
        def get_first_non_empty(series):
            non_empty = series.dropna().map(lambda x: x.strip() if isinstance(x, str) else '').loc[lambda x: x != '']
            if not non_empty.empty:
                print(f"[DEBUG] Propagating features for handle group: {series.name} -> '{non_empty.iloc[0]}'")
                return non_empty.iloc[0]
            else:
                print(f"[DEBUG] No non-empty features found for handle group: {series.name}")
                return ''
        df['Features'] = df.groupby('handle')['Features'].transform(get_first_non_empty)
        empty_count = (df['Features'].isnull() | (df['Features'].map(lambda x: x.strip() if isinstance(x, str) else '') == '')).sum()
        print(f"Rows with empty Features after filling: {empty_count}")

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

    # Clean 'Features' field to handle UTF characters and line spacing
    # (SKIP cleaning Features to preserve key:value and | structure)

    # Keep only necessary columns
    # Ensure the features column is named 'Features' before selecting columns
    if 'features' in df.columns:
        df.rename(columns={'features': 'Features'}, inplace=True)
    columns_to_keep = [
        'handle', 'title', 'Product Type', 'Category', 'tags', 'SKU', 'Weight', 'Better Home Price',
        'Retail Price', 'Description', 'Brand', 'Features', 'Material', 'Returns Policy',
        'Manufacturer URL', 'Warranty', 'url', 'Color', 'Finish', 'Material', 'Style',
        'Image Src', 'Image Alt Text'
    ]
    existing_columns_in_df = [col for col in columns_to_keep if col in df.columns]
    cleaned_df = df[existing_columns_in_df]

    # Debug: Print columns of the final DataFrame before saving
    print("Columns in the final DataFrame:", cleaned_df.columns.tolist())

    # Before saving, remove non-ASCII characters from all string columns
    for col in cleaned_df.select_dtypes(include=['object']).columns:
        cleaned_df[col] = cleaned_df[col].apply(remove_non_ascii)

    # Save the cleaned DataFrame to a new CSV file
    cleaned_df.to_csv(output_file, index=False)
    print(f"Data cleaned and saved to {output_file}")
    
    # Print sample of image sources
    if 'Image Src' in cleaned_df.columns:
        print("\nSample of Image Sources:")
        print(cleaned_df['Image Src'].head())
    else:
        print("\nWarning: Image Src column not found in final output")

    # After renaming columns, handle Product Type logic
    if 'Product Type' in df.columns and 'Category' in df.columns:
        def normalize_product_type(row):
            product_type = row['Product Type']
            category = row['Category']
            # If Product Type is blank or 'Home Appliances', use last segment of Product Category
            if pd.isna(product_type) or str(product_type).strip() == '' or str(product_type).strip().lower() == 'home appliances':
                if isinstance(category, str) and '>' in category:
                    # Use last segment after '>'
                    product_type = category.split('>')[-1].strip()
            # Normalize plural/singular forms
            plural_map = {
                'Air Conditioner': 'Air Conditioners',
                'Air Conditioners': 'Air Conditioners',
                # Add more mappings as needed
            }
            # Map both singular and plural to the plural form
            if isinstance(product_type, str):
                product_type = plural_map.get(product_type, product_type)
            return product_type
        df['Product Type'] = df.apply(normalize_product_type, axis=1)


if __name__ == "__main__":
    input_file = 'products_export_1_4.csv'  # Replace with your input file
    #input_file = 'test.csv'  # Replace with your input file
    output_file = 'cleaned_products_1.4.csv'  # Replace with your desired output file name
    clean_data(input_file, output_file)

