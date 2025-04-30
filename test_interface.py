import streamlit as st
import json
import faiss
import numpy as np
from whatsapp_bot import handle_message, handle_follow_up_question
import os

# Set page config
st.set_page_config(
    page_title="WhatsApp Bot Test Interface",
    page_icon="📱",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'last_query' not in st.session_state:
    st.session_state.last_query = None
if 'last_results' not in st.session_state:
    st.session_state.last_results = None

# Load product terms
with open('product_terms.json', 'r') as f:
    product_terms = json.load(f)

# Load blog embeddings
with open('blog_embeddings.json', 'r') as f:
    blog_data = json.load(f)
    blog_embeddings = blog_data['blog_embeddings']
    blog_metadata = blog_data['metadata']

# Load FAISS index
index = faiss.read_index('blog_faiss_index.index')

# Header
st.title("📱 WhatsApp Bot Test Interface")
st.markdown("---")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Type your message:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Check if this is a follow-up question
            if st.session_state.last_query and st.session_state.last_results:
                response = handle_follow_up_question(
                    prompt,
                    st.session_state.last_query,
                    st.session_state.last_results,
                    product_terms,
                    blog_embeddings,
                    blog_metadata,
                    index
                )
            else:
                response, results = handle_message(
                    prompt,
                    product_terms,
                    blog_embeddings,
                    blog_metadata,
                    index
                )
                st.session_state.last_query = prompt
                st.session_state.last_results = results

            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with status and controls
with st.sidebar:
    st.title("Status")
    
    # Check if required files exist
    if os.path.exists('blog_embeddings.json') and os.path.exists('blog_faiss_index.index'):
        st.success("✅ Blog data loaded successfully")
        st.info(f"📚 {len(blog_metadata)} articles available")
    else:
        st.error("❌ Error loading blog data")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_query = None
        st.session_state.last_results = None
        st.rerun() 