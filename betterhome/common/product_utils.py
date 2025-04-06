"""
Product utilities module for Better Home application.

This module contains functions for handling product-related operations.
"""

import re
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

def find_product_type(query: str, product_terms: Dict[str, Any]) -> Optional[str]:
    """
    Find the most relevant product type for a given query using the product terms dictionary.
    
    Args:
        query: User query
        product_terms: Dictionary containing product terms
        
    Returns:
        Product type or None if not found
    """
    query_lower = query.lower()
    
    # First check exact matches in product terms dictionary
    for product_type, info in product_terms.items():
        if product_type.lower() in query_lower:
            return product_type
            
        # Check categories
        for category in info['categories']:
            if category.lower() in query_lower:
                return product_type
                
    # If no exact match found, try partial matches
    for product_type, info in product_terms.items():
        # Check if any word in the product type is in the query
        product_type_words = product_type.lower().split()
        if any(word in query_lower for word in product_type_words):
            return product_type
            
        # Check categories
        for category in info['categories']:
            category_words = category.lower().split()
            if any(word in query_lower for word in category_words):
                return product_type
    
    return None

def handle_price_query(query: str, df: pd.DataFrame, product_terms: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
    """
    Handle price-based queries.
    
    Args:
        query: User query
        df: DataFrame containing product data
        product_terms: Dictionary containing product terms
        
    Returns:
        DataFrame containing filtered products or None if not a price query
    """
    query_lower = query.lower()
    product_type = None
    is_bldc = False
    
    # Check if the query matches a product type in the product_terms dictionary
    if product_terms:
        product_type = find_product_type(query, product_terms)
        if product_type:
            print(f"Found product type in dictionary: {product_type}")
    
    # If no match in the dictionary, use manual checks
    if not product_type:
        # Check for HDHMR/plywood specific queries
        if 'hdhmr' in query_lower or 'plywood' in query_lower or 'ply' in query_lower:
            product_type = 'Plywood'
        # Check for BLDC specific queries
        elif 'bldc' in query_lower or 'brushless' in query_lower or 'energy efficient fan' in query_lower:
            product_type = 'Ceiling Fan'
            is_bldc = True
        # Check for geyser/water heater in query
        elif 'geyser' in query_lower or 'water heater' in query_lower:
            product_type = 'Water Heater'
        # Check for fan related queries
        elif 'fan' in query_lower or 'ceiling fan' in query_lower:
            product_type = 'Ceiling Fan'
    
    # Filter by product type if specified
    if product_type:
        df = df[df['Product Type'].astype(str) == product_type]
        
        # Further filter for BLDC fans if specified
        if is_bldc:
            df = df[df['title'].str.contains('BLDC|Brushless', case=False, na=False)]
    
    # Sort based on query type
    if 'most expensive' in query_lower or 'highest price' in query_lower:
        sorted_df = df.sort_values('Better Home Price', ascending=False)
        return sorted_df.head(5)
    elif 'least expensive' in query_lower or 'cheapest' in query_lower or 'lowest price' in query_lower:
        sorted_df = df.sort_values('Better Home Price', ascending=True)
        return sorted_df.head(5)
    
    return None

def handle_brand_query(query: str, df: pd.DataFrame, product_terms: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Handle brand-related queries.
    
    Args:
        query: User query
        df: DataFrame containing product data
        product_terms: Dictionary containing product terms
        
    Returns:
        Dictionary containing filtered products and metadata or None if not a brand query
    """
    query_lower = query.lower()
    product_type = None
    
    # Check if it contains warranty-related terms
    is_warranty_query = any(term in query_lower for term in [
        'warranty', 'guarantee', 'guaranty', 'quality', 'life', 'lifetime', 'longevity',
        'years', 'replacement', 'reliable', 'reliability'
    ])
    
    # Check if it's a brand-related query
    is_brand_query = any(term in query_lower for term in [
        'brand', 'brands', 'manufacturer', 'manufacturers', 'make', 'makes', 'company', 'companies', 'list'
    ])
    
    # Also detect patterns like "what brands of X do you have" or "list brands of X"
    if not is_brand_query:
        if re.search(r'what|which|list|show|tell|different|available', query_lower) and re.search(r'of|for', query_lower):
            is_brand_query = True
    
    if not is_brand_query and not is_warranty_query:
        return None
    
    # First try to match product type using the dictionary
    if product_terms:
        product_type = find_product_type(query, product_terms)
        if product_type:
            print(f"Found product type in dictionary for brand query: {product_type}")
    
    # If no match in the dictionary, use manual checks
    if not product_type:
        # Check for HDHMR/plywood specific queries
        if 'hdhmr' in query_lower or 'plywood' in query_lower or 'ply' in query_lower:
            product_type = 'Plywood'
        # Check for product type mentions
        elif 'geyser' in query_lower or 'water heater' in query_lower or 'heater' in query_lower:
            product_type = 'Water Heater'
        elif 'bldc' in query_lower or 'brushless' in query_lower or 'energy efficient fan' in query_lower:
            product_type = 'Ceiling Fan'
            # Specifically filter for BLDC fans
            df = df[df['title'].str.contains('BLDC|Brushless', case=False, na=False)]
        elif 'fan' in query_lower or 'ceiling fan' in query_lower:
            product_type = 'Ceiling Fan'
        elif 'washing machine' in query_lower or 'washer' in query_lower:
            product_type = 'Washing Machine'
        elif 'refrigerator' in query_lower or 'fridge' in query_lower:
            product_type = 'Refrigerator'
        elif 'air conditioner' in query_lower or 'ac' in query_lower:
            product_type = 'Air Conditioner'
    
    if product_type:
        # Filter by product type - ensure both are strings for comparison
        product_type_str = str(product_type)
        filtered_df = df[df['Product Type'].astype(str) == product_type_str]
        
        # If no results, try a partial match
        if len(filtered_df) == 0:
            print(f"No exact matches for '{product_type}', trying partial matches")
            for pt in df['Product Type'].unique():
                pt_str = str(pt).lower()
                product_type_lower = product_type_str.lower()
                if product_type_lower in pt_str or pt_str in product_type_lower:
                    print(f"Found partial match: {pt}")
                    filtered_df = pd.concat([filtered_df, df[df['Product Type'].astype(str) == str(pt)]])
        
        # Get unique brands for this product type
        unique_brands = filtered_df['Brand'].unique()
        print(f"Found {len(unique_brands)} unique brands: {unique_brands}")
        
        # For each brand, get a representative product
        brand_products = []
        for brand in unique_brands:
            brand_df = filtered_df[filtered_df['Brand'] == brand]
            if len(brand_df) > 0:  # Ensure there are products for this brand
                # For warranty queries, prioritize products that mention warranty in the title
                if is_warranty_query:
                    warranty_products = brand_df[brand_df['title'].str.contains('warranty|year', case=False, na=False)]
                    if len(warranty_products) > 0:
                        brand_products.append(warranty_products.iloc[0])  # Use product that mentions warranty
                    else:
                        brand_products.append(brand_df.iloc[0])  # Fallback to first product if none mention warranty
                else:
                    brand_products.append(brand_df.iloc[0])  # Get the first product for each brand
        
        # Create brand products dataframe
        if brand_products:
            result = {
                'dataframe': pd.DataFrame(brand_products),
                'product_type': product_type_str,
                'is_warranty_query': is_warranty_query
            }
            return result
    
    return None

def format_brand_response(products_df: pd.DataFrame, product_type: str, is_warranty_query: bool = False) -> str:
    """
    Format brand response.
    
    Args:
        products_df: DataFrame containing brand products
        product_type: Product type
        is_warranty_query: Whether the query is about warranty
        
    Returns:
        Formatted response string
    """
    # Limit to 3 brands maximum for brevity
    products_df = products_df.head(3)
    
    # Create a concise title
    if is_warranty_query:
        response = f"### {product_type} Brands with Warranty:\n\n"
    else:
        response = f"### {product_type} Brands:\n\n"
    
    # Get total count
    brand_count = len(products_df)
    response += f"{brand_count} brand{'s' if brand_count > 1 else ''} available:\n\n"
    
    for _, product in products_df.iterrows():
        title = product['title']
        brand = product['Brand']
        price = product['Better Home Price']
        retail_price = product.get('Retail Price', 0)
        url = product.get('url', '#')
        
        # Calculate discount percentage if retail price is available
        if retail_price > 0:
            discount = ((retail_price - price) / retail_price) * 100
            discount_text = f"({discount:.1f}% off)"
        else:
            discount_text = ""
        
        # Highlight warranty information if present in the title
        warranty_text = ""
        if is_warranty_query:
            # Extract warranty info from title using regex
            warranty_match = re.search(r'with\s+(\d+)\s+years?\s+warranty', title, re.IGNORECASE)
            if warranty_match:
                warranty_years = warranty_match.group(1)
                warranty_text = f"⭐ {warranty_years} Year Warranty\n"
            elif 'warranty' in title.lower():
                warranty_text = "⭐ Includes Warranty\n"
            elif 'year' in title.lower() and re.search(r'(\d+)\s+years?', title, re.IGNORECASE):
                warranty_years = re.search(r'(\d+)\s+years?', title, re.IGNORECASE).group(1)
                warranty_text = f"⭐ {warranty_years} Year Guarantee\n"
        
        # More concise brand listing
        response += f"**{brand}** {discount_text}\n"
        response += f"₹{price:,.2f}\n"
        if warranty_text:
            response += warranty_text
        # Make the buy link more prominent
        response += f"🛒 [Buy Now]({url})\n\n"
    
    # Add a note about clicking the links
    response += "*Click on 'Buy Now' to purchase the product.*\n"
    
    return response

def format_product_response(products_df: pd.DataFrame) -> str:
    """
    Format product response.
    
    Args:
        products_df: DataFrame containing products
        
    Returns:
        Formatted response string
    """
    # Limit to 3 products maximum for brevity
    products_df = products_df.head(3)
    
    response = "### Products:\n\n"
    
    # Add a special message for BLDC fans if they're in the results
    has_bldc = any(products_df['title'].str.contains('BLDC|Brushless', case=False, na=False))
    if has_bldc:
        response += "💚 **BLDC Fans** - 70% less electricity!\n\n"
    
    for _, product in products_df.iterrows():
        title = product['title']
        price = product['Better Home Price']
        retail_price = product.get('Retail Price', 0)
        url = product.get('url', '#')
        
        # Calculate discount percentage if retail price is available
        if retail_price > 0:
            discount = ((retail_price - price) / retail_price) * 100
            discount_text = f"({discount:.1f}% off)"
        else:
            discount_text = ""
        
        # Add a special highlight for BLDC fans
        is_bldc = 'BLDC' in title or 'Brushless' in title
        energy_label = "💚 " if is_bldc else ""
        
        # More concise product listing
        response += f"**{energy_label}{title}**\n"
        response += f"₹{price:,.2f} {discount_text}\n"
        
        # Add energy saving information for BLDC fans
        if is_bldc:
            response += f"70% energy savings\n"
        
        # Make the buy link more prominent
        response += f"🛒 [Buy Now]({url})\n\n"
    
    # Add a note about clicking the links
    response += "*Click on 'Buy Now' to purchase the product.*\n"
    
    return response

def search_catalog(query: str, df: pd.DataFrame, index: Any, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search catalog for products.
    
    Args:
        query: User query
        df: DataFrame containing product data
        index: FAISS index
        top_k: Number of results to return
        
    Returns:
        List of product dictionaries
    """
    try:
        from betterhome.common.embeddings import get_query_embedding
        
        q_emb = get_query_embedding(query).reshape(1, -1)
        
        # Get the dimension of the FAISS index
        index_dim = index.d
        
        # Ensure query embedding matches index dimension
        if q_emb.shape[1] != index_dim:
            print(f"[Dimension Mismatch] Query dim: {q_emb.shape[1]}, Index dim: {index_dim}")
            # Pad or truncate the query embedding to match index dimension
            if q_emb.shape[1] > index_dim:
                q_emb = q_emb[:, :index_dim]
            else:
                padding = np.zeros((1, index_dim - q_emb.shape[1]), dtype='float32')
                q_emb = np.hstack((q_emb, padding))
        
        # Perform the search with a safe top_k value
        safe_top_k = min(top_k, index.ntotal)
        if safe_top_k == 0:
            print("[Search Error] No vectors in the index")
            return []
            
        D, I = index.search(q_emb, safe_top_k)
        
        # Handle case where no results are found
        if len(I[0]) == 0:
            print("[Search Error] No results found")
            return []
            
        # Get the results, ensuring indices are within bounds
        valid_indices = []
        for i in I[0]:
            if 0 <= i < len(df):
                valid_indices.append(i)
            else:
                print(f"[Search Warning] Index {i} is out of bounds for DataFrame of length {len(df)}")
        
        if not valid_indices:
            print("[Search Error] No valid indices found")
            return []
            
        # Create a list to store valid results
        results = []
        for idx in valid_indices:
            try:
                product = df.iloc[idx].to_dict()
                results.append(product)
            except Exception as e:
                print(f"[Search Error] Error accessing product at index {idx}: {str(e)}")
                continue
        
        return results
        
    except Exception as e:
        print(f"[Search Error] {str(e)}")
        # Return empty list instead of raising exception
        return []

def format_answer(products: List[Dict[str, Any]], query: str) -> str:
    """
    Format answer for products.
    
    Args:
        products: List of product dictionaries
        query: User query
        
    Returns:
        Formatted response string
    """
    if not products:
        return f"I couldn't find any products matching your query: **{query}**. Please try rephrasing your question or check the product name."
    
    try:
        response = f"Found {len(products)} products matching your query: **{query}**\n\n"
        for p in products:
            try:
                title = p.get('title', 'N/A')
                price = p.get('Better Home Price', 'N/A')
                retail = p.get('Retail Price', 'N/A')
                brand = p.get('Brand', 'N/A')
                product_type = p.get('Product Type', 'N/A')
                category = p.get('Category', 'N/A')
                description = p.get('Description', 'N/A')
                sku = p.get('SKU', 'N/A')
                url = p.get('url', '#')
                
                response += f"### {title}\n"
                response += f"- SKU: {sku}\n"
                response += f"- Brand: {brand}\n"
                response += f"- Product Type: {product_type}\n"
                response += f"- Category: {category}\n"
                response += f"- Better Home Price: ₹{price}\n"
                response += f"- Retail Price: ₹{retail}\n"
                response += f"- Description: {description}\n"
                response += f"[Click here to buy]({url})\n\n"
            except Exception as e:
                print(f"[Format Error] Error formatting product: {str(e)}")
                continue
        return response
    except Exception as e:
        print(f"[Format Error] Error in format_answer: {str(e)}")
        return f"I found some products matching your query: **{query}**, but encountered an error while formatting the response. Please try again." 