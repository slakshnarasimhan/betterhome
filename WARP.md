# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Architecture Overview

BetterHome is a comprehensive QA system for home appliances with three main components:

### 1. Core QA System (Root Directory)
- **Product recommendation engine** using semantic search and embeddings
- **Blog content analysis** system with vector search capabilities  
- **WhatsApp bot integration** for customer interactions
- **Vector databases** using FAISS for efficient similarity search
- **Data processing pipeline** for products and blog content

### 2. Web Application (`web_app/`)
- **Flask-based recommendation web app** for generating customized home appliance recommendations
- **Multi-page form interface** collecting detailed user requirements (room-by-room specifications)
- **PDF generation system** creating styled recommendations with product links
- **Static file serving** and upload handling
- **Twilio integration** for OTP verification

### 3. Customer Support (`cs/`)
- **Support agent system** for handling customer queries
- **Knowledge base integration** using embeddings and vector search

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment  
python -m venv betterhome_env
source betterhome_env/bin/activate  # On Windows: betterhome_env\\Scripts\\activate

# Install dependencies (root level)
pip install -r requirements.txt

# Install web app dependencies
pip install -r web_app/requirements.txt

# Install CS module dependencies  
pip install -r cs/requirements.txt

# Set up environment variables
# Create .env file with OPENAI_API_KEY, WHATSAPP_API_KEY, WHATSAPP_API_SECRET
```

### Data Generation & Setup
```bash
# Generate blog embeddings and indices
python process_blogs.py

# Generate product embeddings
python generate-embedding.py

# Alternative embedding generation with OpenAI
python generate-embedding-openai.py

# Set up web app static files
cd web_app && bash setup.sh
```

### Running Applications

#### QA System & Testing
```bash
# Start Streamlit test interface
streamlit run test_interface.py

# Run QA system with corrected logic
streamlit run ask-questions-updated-corrected.py --server.port=8501 --server.address=0.0.0.0

# Start WhatsApp bot server
python whatsapp_bot.py

# Start FastAPI server
python betterhome_api_server.py
```

#### Web Application
```bash
# Set up and run Flask web app
cd web_app
export FLASK_APP=app.py
export FLASK_ENV=production
flask run --host=0.0.0.0 --port=5002

# Alternative: Run directly
python app.py
```

### Testing & Validation

#### Single Tests
```bash
# Test individual components
python simple_test.py
python test_interface.py
python test_blog_extraction.py
python test_bestseller.py

# Test QA via different interfaces
python test_qa_streamlit.py
python test_qa_via_api.py
```

#### Generate & Run Test Suites
```bash
# Generate test questions
python regenerate_test_questions.py

# Analyze QA results and generate diagnostics
python analyze_and_patch_qa_results.py

# Run blog processing tests
python test_blog_search.py

# WhatsApp bot testing
python test-whatsapp-bot.py
```

#### Web App Testing
```bash
# Automated form testing with Excel data
cd web_app
python test_automation.py <input_excel_file>

# Generate test data
python test_generation.py
```

## Key Architecture Patterns

### Embedding & Vector Search Pipeline
1. **Content Processing**: Blog content and product data are processed and cleaned
2. **Embedding Generation**: Text is converted to vector embeddings using OpenAI API or local models
3. **Index Creation**: FAISS indices are built for efficient similarity search
4. **Query Processing**: User queries are embedded and matched against existing indices
5. **Result Ranking**: Results are ranked by similarity and filtered for relevance

### Multi-Modal Data Handling
- **Product Catalog**: CSV/JSON data with product specifications and metadata
- **Blog Content**: Web scraping with YouTube transcript extraction
- **User Requirements**: Structured form data with room-by-room specifications
- **Knowledge Graph**: NetworkX-based product relationship mapping

### Service Architecture
- **Stateless APIs**: FastAPI and Flask apps with JSON interfaces
- **Background Processing**: Separate scripts for data generation and updates
- **File-based Persistence**: JSON and FAISS index files for data storage
- **External Integrations**: WhatsApp, Twilio, OpenAI API connections

## Important Files & Dependencies

### Core Data Files (Must Exist)
- `product_terms.json` - Product categories and synonyms mapping
- `blog_embeddings.json` - Processed blog content embeddings  
- `embeddings.json` - Product embeddings for similarity search
- `faiss_index.index_*` - FAISS indices for different content types
- `blog_faiss_index.index` - Blog-specific search index
- `product_catalog.json` - Main product database

### Configuration Files
- `.env` - API keys and environment variables
- `home_config.yaml` / `home_appliance_config.yaml` - Product configuration
- `budget_config.yaml` - Budget tier configurations
- `feature_priorities_config.json` - Product feature prioritization

## Development Guidelines

### Working with Embeddings
- Always regenerate embeddings after significant data changes
- Use `generate-embedding-openai.py` for production-quality embeddings
- Test embedding quality with `test-blog-embedding.py`
- FAISS indices must be rebuilt after embedding updates

### Web App Development
- Flask app uses dynamic imports for recommendation generation
- Templates are in `web_app/templates/` with auto-reload enabled
- Static assets require proper setup via `setup.sh`
- File uploads go to timestamped directories in `uploads/`

### Testing Strategy
- Use Streamlit interfaces for interactive testing and debugging
- Excel-driven automation for comprehensive web app testing  
- WhatsApp bot testing requires manual QR code scanning
- API testing uses both direct calls and Streamlit wrappers

### Data Processing
- Blog processing includes web scraping and YouTube transcript extraction
- Product data requires cleaning and categorization before embedding
- Knowledge graphs help with product relationship discovery
- Synonym dictionaries improve query matching accuracy
