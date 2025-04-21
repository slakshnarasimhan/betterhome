# BetterHome QA System

A comprehensive question-answering system for home appliances and products, featuring blog content analysis, WhatsApp integration, and a Streamlit test interface.

## New Features and Updates

### PDF Generation
- The `create_styled_pdf` function generates a PDF document with product recommendations.
- Features include clickable product links and proper currency formatting using the DejaVuSans font.

### Product Recommendations
- The system generates specific product recommendations based on user data.
- Includes logic for handling washing machine recommendations and other appliances.

### Debugging and Logging
- Added debug statements to trace the flow of data and identify issues in the recommendation logic.
- Useful for diagnosing problems with product recommendations and PDF generation.

## Project Structure

### Core Components
- `process_blogs.py` - Crawls and processes blog content, extracts YouTube transcripts, and generates blog embeddings
- `whatsapp_bot.py` - WhatsApp integration for product queries and customer interactions
- `test_interface.py` - Streamlit interface for testing the QA system with a chat-like interface
- `ask-questions-updated-corrected.py` - Core QA logic and product matching using embeddings
- `generate-embedding.py` - Generates embeddings for products and content using OpenAI API
- `analyze_and_patch_qa_results.py` - Analyzes QA test results and generates diagnostics

### Data Processing
- `process-catalog.py` - Processes and cleans the product catalog data
- `generate-embedding-new.py` - Alternative embedding generation script with updated logic
- `generate-vector-db.py` - Creates vector database for efficient similarity search
- `generate-blog-embeddings.py` - Specialized script for generating blog content embeddings

### Testing and Validation
- `test_qa_streamlit.py` - Tests the QA system through Streamlit interface
- `test_qa_via_api.py` - Tests the QA system through API endpoints
- `test_interface.py` - Streamlit interface for manual testing
- `test_streamlit.py` - Basic Streamlit test script
- `simple_test.py` - Minimal test script for Streamlit functionality
- `regenerate_test_questions.py` - Generates new test questions for QA validation

### API and Server
- `betterhome_api_server.py` - FastAPI server for the QA system
- `start_streamlit.sh` - Shell script to start the Streamlit server

### Development and Analysis
- `ollama.ipynb` - Jupyter notebook for Ollama model experimentation
- `test-blog-embedding.py` - Tests blog embedding generation
- `test.py` - General testing script

### Configuration Files
- `.env` - Environment variables and API keys
- `requirements.txt` - Python package dependencies
- `product_terms.json` - Product categories and synonyms
- `synonym_dict.json` - Product synonym mappings
- `updated_synonym_dict.json` - Updated product synonyms

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Git
- Ollama (for local LLM support)
- Chrome/Chromium (for web scraping)

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd betterhome
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   python -m venv betterhome_env
   source betterhome_env/bin/activate  # On Windows: betterhome_env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   Create a `.env` file in the project root with:
   ```
   OPENAI_API_KEY=your_openai_api_key
   WHATSAPP_API_KEY=your_whatsapp_api_key
   WHATSAPP_API_SECRET=your_whatsapp_api_secret
   ```

5. **Install and Start Ollama**
   ```bash
   # Install Ollama (instructions vary by OS)
   # Start Ollama service
   ollama serve
   ```

6. **Generate Initial Data**
   ```bash
   # Process blog content and generate embeddings
   python process_blogs.py
   
   # Generate product embeddings
   python generate-embedding.py
   ```

## Running the Application

1. **Start the Streamlit Test Interface**
   ```bash
   streamlit run test_interface.py
   ```
   Access the interface at `http://localhost:8501`

2. **Run the WhatsApp Bot**
   ```bash
   python whatsapp_bot.py
   ```

3. **Test the QA System**
   ```bash
   python test_qa_streamlit.py
   ```

## Testing and Diagnostics

1. **Generate Test Questions**
   ```bash
   python regenerate_test_questions.py
   ```

2. **Run QA Tests**
   ```bash
   python test_qa_via_api.py
   ```

3. **Analyze Results**
   ```bash
   python analyze_and_patch_qa_results.py
   ```

## File Dependencies

The system requires several key files to function:

- `product_terms.json` - Product categories and synonyms
- `blog_embeddings.json` - Processed blog content embeddings
- `blog_faiss_index.index` - FAISS index for blog content
- `embeddings.json` - Product embeddings
- `faiss_index.index_*` - Various FAISS indices for product matching

## Troubleshooting

1. **Missing Dependencies**
   - Ensure all requirements are installed: `pip install -r requirements.txt`
   - Check Python version: `python --version`

2. **API Issues**
   - Verify API keys in `.env` file
   - Check API rate limits and quotas

3. **Ollama Issues**
   - Ensure Ollama service is running
   - Check model availability: `ollama list`

4. **Data Generation Issues**
   - Run `process_blogs.py` to regenerate blog data
   - Run `generate-embedding.py` to regenerate product embeddings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license information here] 