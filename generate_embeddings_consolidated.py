#!/usr/bin/env python3
"""
Consolidated embedding generation script for the BetterHome project.
This script replaces multiple duplicate embedding generation scripts and provides
a unified interface for generating embeddings using different providers.

Usage:
    python generate_embeddings_consolidated.py --provider openai --csv cleaned_products.csv
    python generate_embeddings_consolidated.py --provider ollama --model llama2
"""

import argparse
import sys
import os
import numpy as np
from tqdm import tqdm

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_utils import (
    load_product_catalog, 
    save_json_file, 
    build_faiss_index
)
from utils.embedding_utils import (
    create_embedding_provider,
    prepare_product_entries,
    generate_embeddings_batch,
    validate_embeddings
)


def main():
    parser = argparse.ArgumentParser(description='Generate embeddings for BetterHome products')
    parser.add_argument('--provider', choices=['openai', 'ollama'], default='openai',
                       help='Embedding provider to use')
    parser.add_argument('--model', default=None,
                       help='Model name (openai: text-embedding-3-small, ollama: llama2)')
    parser.add_argument('--csv', default='cleaned_products.csv',
                       help='Path to product catalog CSV file')
    parser.add_argument('--output', default='embeddings.json',
                       help='Output file for embeddings')
    parser.add_argument('--index-output', default='faiss_index.index',
                       help='Output file for FAISS index')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for processing')
    parser.add_argument('--enhanced-features', action='store_true',
                       help='Include enhanced feature extraction')
    parser.add_argument('--api-key', default=None,
                       help='API key for OpenAI (if not in environment)')
    parser.add_argument('--ollama-url', default='http://localhost:11434/api/embeddings',
                       help='Ollama API endpoint')
    
    args = parser.parse_args()
    
    print("🏠 BetterHome Embedding Generation Tool")
    print(f"Provider: {args.provider}")
    print(f"CSV File: {args.csv}")
    print(f"Output: {args.output}")
    print("-" * 50)
    
    # Step 1: Load product catalog
    print("📂 Loading product catalog...")
    df = load_product_catalog(args.csv)
    if df is None or df.empty:
        print("❌ Error: Could not load product catalog.")
        sys.exit(1)
    
    print(f"✅ Loaded {len(df)} products")
    
    # Step 2: Prepare entries for embedding
    print("📝 Preparing product entries...")
    entries = prepare_product_entries(df, include_enhanced_features=args.enhanced_features)
    
    if not entries:
        print("❌ Error: No product entries prepared.")
        sys.exit(1)
    
    print(f"✅ Prepared {len(entries)} product entries")
    
    # Step 3: Create embedding provider
    print(f"🤖 Initializing {args.provider} embedding provider...")
    provider_kwargs = {}
    
    if args.provider == 'openai':
        if args.model:
            provider_kwargs['model'] = args.model
        if args.api_key:
            provider_kwargs['api_key'] = args.api_key
    elif args.provider == 'ollama':
        if args.model:
            provider_kwargs['model'] = args.model
        provider_kwargs['url'] = args.ollama_url
    
    try:
        provider = create_embedding_provider(args.provider, **provider_kwargs)
        print(f"✅ {args.provider} provider initialized")
    except Exception as e:
        print(f"❌ Error initializing provider: {e}")
        sys.exit(1)
    
    # Step 4: Generate embeddings
    print("🔄 Generating embeddings...")
    embeddings = generate_embeddings_batch(entries, provider, args.batch_size)
    
    if not embeddings or all(e is None for e in embeddings):
        print("❌ Error: No embeddings generated.")
        sys.exit(1)
    
    # Filter out None embeddings
    valid_embeddings = [e for e in embeddings if e is not None]
    valid_entries = [entries[i] for i, e in enumerate(embeddings) if e is not None]
    
    print(f"✅ Generated {len(valid_embeddings)} valid embeddings")
    
    # Step 5: Validate embeddings
    print("🔍 Validating embeddings...")
    if validate_embeddings(valid_embeddings):
        print("✅ All embeddings are valid")
    else:
        print("⚠️ Warning: Some embeddings may be invalid")
    
    # Step 6: Save embeddings
    print(f"💾 Saving embeddings to {args.output}...")
    embeddings_data = {
        'product_embeddings': [e.tolist() for e in valid_embeddings],
        'entries': valid_entries,
        'metadata': {
            'provider': args.provider,
            'model': args.model or f"default-{args.provider}",
            'total_products': len(df),
            'valid_embeddings': len(valid_embeddings),
            'embedding_dimension': len(valid_embeddings[0]) if valid_embeddings else 0,
            'enhanced_features': args.enhanced_features
        }
    }
    
    if save_json_file(embeddings_data, args.output):
        print(f"✅ Embeddings saved to {args.output}")
    else:
        print(f"❌ Error saving embeddings to {args.output}")
        sys.exit(1)
    
    # Step 7: Build and save FAISS index
    print(f"🏗️ Building FAISS index...")
    embeddings_array = np.array(valid_embeddings)
    index = build_faiss_index(embeddings_array, args.index_output)
    
    if index:
        print(f"✅ FAISS index saved to {args.index_output}")
    else:
        print("❌ Error building FAISS index")
        sys.exit(1)
    
    # Step 8: Summary
    print("\n🎉 Embedding generation completed successfully!")
    print(f"📊 Summary:")
    print(f"  • Total products processed: {len(df)}")
    print(f"  • Valid embeddings generated: {len(valid_embeddings)}")
    print(f"  • Embedding dimension: {len(valid_embeddings[0])}")
    print(f"  • Provider: {args.provider}")
    print(f"  • Enhanced features: {args.enhanced_features}")
    print(f"  • Output files:")
    print(f"    - Embeddings: {args.output}")
    print(f"    - FAISS index: {args.index_output}")


if __name__ == "__main__":
    main()