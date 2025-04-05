# BetterHome QA System

A comprehensive question-answering system for home appliances and products, featuring blog content analysis, WhatsApp integration, and a Streamlit test interface.

## Project Structure

- `process_blogs.py` - Crawls and processes blog content, extracts YouTube transcripts
- `whatsapp_bot.py` - WhatsApp integration for product queries
- `test_interface.py` - Streamlit interface for testing the QA system
- `ask-questions-updated-corrected.py` - Core QA logic and product matching
- `generate-embedding.py` - Generates embeddings for products and content
- `analyze_and_patch_qa_results.py` - Analyzes QA test results and generates diagnostics

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