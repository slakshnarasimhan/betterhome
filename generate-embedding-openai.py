import openai
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import faiss
from typing import Dict, List, Any
import unicodedata
import re

# ==========================
# Configuration
# ==========================
MODEL_NAME = "text-embedding-3-small"
CSV_FILE_PATH = 'cleaned_products_1.4.csv'
EMBEDDINGS_FILE_PATH = 'embeddings.json'
PRODUCT_CATALOG_PATH = 'web_app/product_catalog.json'

# FAISS index output files
"""
INDEX_FILE_PRODUCT = 'faiss_index.index_product'
INDEX_FILE_TYPE = 'faiss_index.index_type'
INDEX_FILE_BRAND = 'faiss_index.index_brand'
INDEX_FILE_IMAGE = 'faiss_index.index_image'
"""

# ==========================
# Step 1: Load Product Catalog
# ==========================
def load_product_catalog(file_path):
    df = pd.read_csv(file_path)
    print(f"Successfully loaded product catalog with {len(df)} entries.")
    
    # Debug: Print detailed column information
    print("\nAll columns in CSV file:")
    for col in df.columns:
        print(f"Column: '{col}'")
        print(f"Sample values: {df[col].head().tolist()}")
        print(f"Null count: {df[col].isnull().sum()}")
        print("---")
    
    # Check specifically for Product Type column variations
    possible_product_type_cols = [col for col in df.columns if 'product' in col.lower() and 'type' in col.lower()]
    if possible_product_type_cols:
        print("\nPossible Product Type columns found:")
        for col in possible_product_type_cols:
            print(f"\nColumn: '{col}'")
            print(f"Sample values: {df[col].head().tolist()}")
            print(f"Unique values: {df[col].unique().tolist()}")
    else:
        print("\nNo columns containing 'product' and 'type' found!")
    
    df['Better Home Price'] = pd.to_numeric(df['Better Home Price'], errors='coerce')
    df['Retail Price'] = pd.to_numeric(df['Retail Price'], errors='coerce')
    
    # Ensure 'Features' is included
    if 'Features (product.metafields.custom.features)' in df.columns:
        df.rename(columns={'Features (product.metafields.custom.features)': 'Features'}, inplace=True)

    # Fill missing 'Features' for rows with the same 'Handle'
    if 'Handle' in df.columns and 'Features' in df.columns:
        for handle, group in df.groupby('Handle'):
            if group['Features'].isnull().any():
                filled_value = group['Features'].dropna().iloc[0] if not group['Features'].dropna().empty else 'Not Available'
                df.loc[df['Handle'] == handle, 'Features'] = df.loc[df['Handle'] == handle, 'Features'].fillna(filled_value)

    return df

"""
# ==========================
# Step 2: Prepare Entries
# ==========================
def prepare_entries(df):
    entries = []
    product_type_entries = []
    brand_entries = []
    image_entries = []
    
    for _, row in df.iterrows():
        if pd.notnull(row.get('Retail Price')) and row['Retail Price'] > 0:
            discount_percentage = ((row['Retail Price'] - row['Better Home Price']) / row['Retail Price']) * 100
            discount_text = f"Better Home Price is {discount_percentage:.2f}% less than Retail Price."
        else:
            discount_text = "No discount available."

        entry = (
            f"Product Type: {row.get('Product Type', 'Not Available')}. "
            f"Brand: {row.get('Brand', 'Not Available')}. "
            f"Title: {row.get('title', 'Not Available')}. "
            f"Better Home Price: {row.get('Better Home Price', 'Not Available')} INR. "
            f"Retail Price: {row.get('Retail Price', 'Not Available')} INR. "
            f"{discount_text} "
            f"Warranty: {row.get('Warranty', 'Not Available')}. "
            f"Features: {row.get('Features', 'Not Available')}. "
            f"Description: {row.get('Description', 'Not Available')}. "
            f"Product URL: {row.get('url', 'Not Available')}."
        )
        entries.append(entry)
        
        # Create image entry only if image source exists
        if pd.notna(row.get('Image Src')):
            image_entry = (
                f"Product: {row.get('title', 'Not Available')}. "
                f"Brand: {row.get('Brand', 'Not Available')}. "
                f"Type: {row.get('Product Type', 'Not Available')}. "
                f"Image Source: {row.get('Image Src')}"
            )
            image_entries.append(image_entry)

        # Product type entry
        if pd.notna(row.get('Product Type')):
            product_type_entries.append(f"Product Type: {row.get('Product Type')}")
        
        # Brand entry
        if pd.notna(row.get('Brand')):
            brand_entries.append(f"Brand: {row.get('Brand')}")

    return entries, product_type_entries, brand_entries, image_entries
"""

# ==========================
# Step 3: Save Product Catalog
# ==========================
def convert_to_cm(value: float, unit: str) -> float:
    """Convert various length units to centimeters"""
    unit = unit.lower() if unit else ''
    if unit in ['mm', 'millimeter', 'millimeters']:
        return value / 10
    elif unit in ['m', 'meter', 'meters']:
        return value * 100
    elif unit in ['inch', 'inches', '"']:
        return value * 2.54
    return value  # Assume cm if no unit or unrecognized unit

def parse_features(features_str: str) -> Dict[str, Any]:
    """Convert features string (separated by '|') into a structured dictionary of features."""
    features_dict = {
        'raw_features': [],  # Keep original feature strings
        'parsed_features': {},  # Structured key-value pairs
        'numeric_features': {},  # Features with numeric values
        'boolean_features': {},  # Yes/No type features
        'text_features': {}  # Text-based features
    }
    
    # Handle potential non-string types gracefully
    if not isinstance(features_str, str) or pd.isna(features_str):
        return features_dict

    # Split by '|' and process each feature
    feature_items = [f.strip() for f in features_str.split('|') if f.strip()]
    
    for feature in feature_items:
        # Store raw feature
        features_dict['raw_features'].append(feature)
        
        # Try to split into key-value pair
        if ':' in feature:
            key, value = [part.strip() for part in feature.split(':', 1)]
            # Special handling for Room Type
            if key.lower() == 'room type':
                if ',' not in value:
                    # Use regex to split before each capital letter that starts a word (except the first)
                    value = re.sub(r'(?<!^)(?=[A-Z][a-z])', ', ', value)
                    value = ', '.join([v.strip() for v in value.split(',')])
            features_dict['parsed_features'][key] = value
            # Try to convert to number
            try:
                # Extract numeric value and unit if present
                numeric_match = re.match(r'([\d.]+)\s*([a-zA-Z°%"]+)?', value)
                if numeric_match:
                    numeric_value = float(numeric_match.group(1))
                    unit = numeric_match.group(2) if numeric_match.group(2) else None
                    features_dict['numeric_features'][key] = {
                        'value': numeric_value,
                        'unit': unit,
                        'raw': value
                    }
                # Check for boolean-like values
                elif value.lower() in ['yes', 'no', 'true', 'false']:
                    features_dict['boolean_features'][key] = value.lower() in ['yes', 'true']
                else:
                    # Store as text feature
                    features_dict['text_features'][key] = value
            except ValueError:
                # If conversion fails, store as text feature
                features_dict['text_features'][key] = value
        else:
            # Handle features without key-value format
            features_dict['text_features'][feature.lower()] = True

    return features_dict

def standardize_fan_measurements(features_dict: Dict[str, Any], product_type: str, title: str = None, description: str = None) -> Dict[str, Any]:
    """Standardize fan measurements, particularly blade length/fan length to cm"""
    if product_type.lower() != 'ceiling fan':
        return features_dict
    
    # Initialize standard fan measurements
    blade_length_cm = None
    blade_length_key_found = None
    
    # Try to extract length from title first if provided
    if blade_length_cm is None and title:
        match = re.search(r'(\d+)\s*(mm|cm|m|inch|")', title.lower())
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            blade_length_cm = convert_to_cm(value, unit)
            blade_length_key_found = 'title'
    
    # List of possible keys for blade/fan length
    length_keys = ['blade length', 'fan length', 'sweep size', 'sweep', 'size', 'fan size', 'sweep length', 'length', 'diameter']
    
    # First check numeric features
    if blade_length_cm is None:
        for key in features_dict.get('numeric_features', {}):
            if any(length_key in key.lower() for length_key in length_keys):
                value = features_dict['numeric_features'][key]['value']
                unit = features_dict['numeric_features'][key]['unit']
                blade_length_cm = convert_to_cm(value, unit)
                blade_length_key_found = key
                break
    
    # If not found in numeric features, check parsed features
    if blade_length_cm is None:
        for key in features_dict.get('parsed_features', {}):
            if any(length_key in key.lower() for length_key in length_keys):
                value_str = features_dict['parsed_features'][key]
                if isinstance(value_str, str):
                    match = re.search(r'(\d+)\s*(mm|cm|m|inch|")', value_str.lower())
                    if match:
                        value = float(match.group(1))
                        unit = match.group(2)
                        blade_length_cm = convert_to_cm(value, unit)
                        blade_length_key_found = key
                        break
                    match = re.match(r'(\d+)', value_str)
                    if match:
                        value = float(match.group(1))
                        blade_length_cm = value  # Assume cm if no unit specified
                        blade_length_key_found = key
                        break
    
    # If still not found, try to extract from raw features
    if blade_length_cm is None:
        size_features = [f for f in features_dict.get('raw_features', []) if any(key in f.lower() for key in length_keys)]
        for feature in size_features:
            match = re.search(r'(\d+)\s*(mm|cm|m|inch|")', feature.lower())
            if match:
                value = float(match.group(1))
                unit = match.group(2)
                blade_length_cm = convert_to_cm(value, unit)
                blade_length_key_found = feature
                break
            match = re.search(r'(?:' + '|'.join(length_keys) + r')\s*(?::|is|of)?\s*(\d+)', feature.lower())
            if match:
                value = float(match.group(1))
                blade_length_cm = value  # Assume cm if no unit specified
                blade_length_key_found = feature
                break
    
    # If still not found, try to extract from description
    if blade_length_cm is None and description:
        size_patterns = [
            r'(\d+)\s*(mm|cm|m|inch|")',
            r'sweep\s+(?:size|length)?\s*(?:of|:)?\s*(\d+)\s*(mm|cm|m|inch|")',
            r'(\d+)\s*(mm|cm|m|inch|")\s*(?:sweep|blade|fan)',
            r'(?:sweep|blade|fan)\s*(?:size|length)?\s*(?:of|:)?\s*(\d+)\s*(mm|cm|m|inch|")',
            r'(?:sweep|blade|fan)\s*(?:size|length)?\s*(?:of|:)?\s*(\d+)',
            r'(\d+)\s*(?:mm|cm|m|inch|")?(?:\s+(?:' + '|'.join(length_keys) + r'))',
        ]
        for pattern in size_patterns:
            match = re.search(pattern, description.lower())
            if match:
                value_groups = [g for g in match.groups() if g and g.replace('.', '').isdigit()]
                if value_groups:
                    value = float(value_groups[0])
                    unit_groups = [g for g in match.groups() if g and not g.replace('.', '').isdigit()]
                    unit = unit_groups[0] if unit_groups else 'mm'
                    blade_length_cm = convert_to_cm(value, unit)
                    blade_length_key_found = 'description'
                    break
    
    # --- Always set numeric_features['blade_length'] if any size key is found ---
    if blade_length_cm is not None:
        if 60 <= blade_length_cm <= 200:
            features_dict['numeric_features']['blade_length'] = {
                'value': round(blade_length_cm, 2),
                'unit': 'cm',
                'raw': f"{round(blade_length_cm, 2)} cm"
            }
            features_dict['parsed_features']['blade_length'] = f"{round(blade_length_cm, 2)} cm"
            # Remove any old inconsistent keys
            for key in list(features_dict['numeric_features'].keys()):
                if any(length_key in key.lower() for length_key in length_keys) and key != 'blade_length':
                    del features_dict['numeric_features'][key]
            for key in list(features_dict['parsed_features'].keys()):
                if any(length_key in key.lower() for length_key in length_keys) and key != 'blade_length':
                    del features_dict['parsed_features'][key]
    else:
        # If a size key is found but out of range, still set blade_length for debugging
        for key in features_dict.get('numeric_features', {}):
            if any(length_key in key.lower() for length_key in length_keys):
                features_dict['numeric_features']['blade_length'] = features_dict['numeric_features'][key]
                break
    return features_dict

def save_product_catalog(df, file_path=PRODUCT_CATALOG_PATH):
    def clean_text(text):
        if not isinstance(text, str):
            return text
        # Normalize unicode characters and remove control characters
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.category(c).startswith('C'))
        
        # Remove specific problematic characters
        text = text.replace('\u200e', '')  # Remove LEFT-TO-RIGHT MARK
        text = text.replace('\u200f', '')  # Remove RIGHT-TO-LEFT MARK
        text = text.replace('\u200b', '')  # Remove ZERO WIDTH SPACE
        
        # Convert to ASCII, replacing non-ASCII with closest equivalent
        text = text.encode('ascii', 'ignore').decode('ascii')
        return text.strip()

    # Debug: Print feature information before processing
    print("\nFeature information before processing:")
    print(f"Total rows: {len(df)}")
    print(f"Null features count: {df['Features'].isnull().sum() if 'Features' in df.columns else 'Features column not found'}")
    
    # Track ceiling fan statistics
    ceiling_fan_count = 0
    fans_with_blade_length = 0
    fans_missing_length = []
    
    if 'Features' in df.columns:
        print("Sample features:")
        print(df['Features'].head())
        # Count empty feature lists
        empty_features = df['Features'].apply(lambda x: not bool(parse_features(x)['raw_features']) if pd.notna(x) else True)
        print(f"Rows with empty features: {empty_features.sum()}")

    catalog = []
    empty_features_count = 0
    feature_keys_stats = {}  # Track frequency of feature keys
    
    for _, row in df.iterrows():
        product_type = clean_text(row.get('Product Type', 'Not Available'))
        # Rename 'Bath Fittings' to 'Bathroom Fittings'
        if product_type == 'Bath Fittings':
            product_type = 'Bathroom Fittings'
        title = clean_text(row.get('title', 'Not Available'))
        description = clean_text(row.get('Description', ''))
        
        # Parse and clean features
        features = parse_features(row.get('Features', ''))
        
        # Standardize measurements for ceiling fans
        if product_type.lower() == 'ceiling fan':
            ceiling_fan_count += 1
            features = standardize_fan_measurements(features, product_type, title, description)
            if 'blade_length' in features['numeric_features']:
                fans_with_blade_length += 1
            else:
                # Store information about fans missing length
                fans_missing_length.append({
                    'title': title,
                    'description': description,
                    'raw_features': features['raw_features'],
                    'parsed_features': features['parsed_features'],
                    'numeric_features': features['numeric_features']
                })
        
        # Update feature keys statistics
        for key in features['parsed_features'].keys():
            feature_keys_stats[key] = feature_keys_stats.get(key, 0) + 1
        
        if not features['raw_features']:
            empty_features_count += 1
            if empty_features_count <= 5:  # Print first 5 examples of empty features
                print(f"\nEmpty features for row:")
                print(f"SKU: {row.get('SKU', 'N/A')}")
                print(f"Title: {title}")
                print(f"Raw features value: {row.get('Features', 'N/A')}")

        # Check if product is a best seller by looking at Tags
        tags = str(row.get('tags', '')).split(',')
        is_best_seller = 'Best Seller' in [tag.strip() for tag in tags]

        product = {
            'sku': clean_text(row.get('SKU', 'Not Available')),
            'product_type': product_type,
            'brand': clean_text(row.get('Brand', 'Not Available')),
            'title': title,
            'better_home_price': row.get('Better Home Price', 'Not Available'),
            'retail_price': row.get('Retail Price', 'Not Available'),
            'warranty': clean_text(row.get('Warranty', 'Not Available')),
            'features': features,  # Now contains structured feature information
            'description': description,
            'url': clean_text(row.get('url', 'Not Available')),
            'image_src': clean_text(row.get('Image Src', 'Not Available')),
            'best_seller': 'Yes' if is_best_seller else 'No'
        }
        catalog.append(product)

    # Print feature keys statistics
    print("\nFeature keys statistics:")
    sorted_features = sorted(feature_keys_stats.items(), key=lambda x: x[1], reverse=True)
    print("Top 20 most common feature keys:")
    for key, count in sorted_features[:20]:
        print(f"{key}: {count} occurrences")

    print(f"\nCeiling Fan Statistics:")
    print(f"Total ceiling fans: {ceiling_fan_count}")
    print(f"Fans with standardized blade length: {fans_with_blade_length}")
    print(f"Fans missing blade length: {ceiling_fan_count - fans_with_blade_length}")
    
    print("\nExamining fans missing blade length:")
    for i, fan in enumerate(fans_missing_length, 1):
        print(f"\nFan {i}:")
        print(f"Title: {fan['title']}")
        print(f"Description: {fan['description']}")
        print("Raw features:", fan['raw_features'])
        print("Parsed features:", fan['parsed_features'])
        print("Numeric features:", fan['numeric_features'])
    
    # Debug: Print sample of ceiling fan entries
    print("\nSample of ceiling fan entries with standardized measurements (first 2):")
    ceiling_fans = [p for p in catalog if p['product_type'].lower() == 'ceiling fan'][:2]
    for i, fan in enumerate(ceiling_fans):
        print(f"\nCeiling Fan {i+1}:")
        print(f"Title: {fan['title']}")
        print("Features structure:")
        print("- Numeric features:", json.dumps(fan['features']['numeric_features'], indent=2))
        print("- Boolean features:", json.dumps(fan['features']['boolean_features'], indent=2))
        print("- Text features:", json.dumps(fan['features']['text_features'], indent=2))
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump({'products': catalog}, f, indent=2, ensure_ascii=True)
    print(f"\nProduct catalog saved successfully to {file_path}.")

"""
# ==========================
# Step 4: Generate Embeddings using OpenAI
# ==========================
def generate_openai_embeddings(entries, batch_size=10):
    embeddings = []
    openai.api_key = st.secrets["OPENAI_API_KEY"]

    def generate_batch(batch):
        try:
            response = openai.Embedding.create(
                model=MODEL_NAME,
                input=batch
            )
            return [e.embedding for e in response.data]
        except Exception as e:
            print(f"Error generating batch embeddings: {str(e)}")
            return []

    with ThreadPoolExecutor() as executor:
        batches = [entries[i:i + batch_size] for i in range(0, len(entries), batch_size)]
        results = list(tqdm(executor.map(generate_batch, batches), total=len(batches), desc="Generating Embeddings"))
        for batch_embeddings in results:
            embeddings.extend(batch_embeddings)

    return embeddings

# ==========================
# Step 5: Save Embeddings
# ==========================
def save_embeddings(embeddings_dict, file_name):
    with open(file_name, 'w') as f:
        json.dump(embeddings_dict, f)
    print(f"Embeddings saved successfully to {file_name}.")

# ==========================
# Step 6: Build FAISS Index
# ==========================
def build_faiss_index(embeddings, index_file_path):
    if not embeddings:
        print("Error: No embeddings to build the index.")
        return None

    try:
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        faiss.write_index(index, index_file_path)
        print(f"FAISS index saved at {index_file_path}")
        return index
    except Exception as e:
        print(f"Error building FAISS index: {str(e)}")
        return None
"""

# ==========================
# Main Execution
# ==========================
def main():
    print("Starting main function...")
    print(f"Looking for CSV file at: {CSV_FILE_PATH}")
    
    df = load_product_catalog(CSV_FILE_PATH)
    if df.empty:
        print("Product catalog could not be loaded. Exiting.")
        return

    # Save the product catalog with image URLs
    print("Saving product catalog...")
    save_product_catalog(df)

    """
    # Prepare all entries
    print("Preparing entries...")
    product_entries, product_type_entries, brand_entries, image_entries = prepare_entries(df)
    
    # Generate embeddings for all types
    print("Generating embeddings...")
    product_embeddings = generate_openai_embeddings(product_entries)
    product_type_embeddings = generate_openai_embeddings(product_type_entries)
    brand_embeddings = generate_openai_embeddings(brand_entries)
    image_embeddings = generate_openai_embeddings(image_entries)

    # Save all embeddings to one file
    embeddings_dict = {
        'product_embeddings': product_embeddings,
        'product_type_embeddings': product_type_embeddings,
        'brand_embeddings': brand_embeddings,
        'image_embeddings': image_embeddings,
        'metadata': {
            'total_products': len(df),
            'products_with_images': len(image_entries),
            'unique_product_types': df['Product Type'].nunique(),
            'unique_brands': df['Brand'].nunique()
        }
    }
    save_embeddings(embeddings_dict, EMBEDDINGS_FILE_PATH)

    # Build FAISS indexes
    print("Building FAISS indexes...")
    build_faiss_index(product_embeddings, INDEX_FILE_PRODUCT)
    build_faiss_index(product_type_embeddings, INDEX_FILE_TYPE)
    build_faiss_index(brand_embeddings, INDEX_FILE_BRAND)
    build_faiss_index(image_embeddings, INDEX_FILE_IMAGE)
    """

    print("\nProduct catalog generation completed successfully!")

if __name__ == "__main__":
    main()

