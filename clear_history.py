import streamlit as st

# Clear all session state variables
if 'conversation_history' in st.session_state:
    st.session_state['conversation_history'] = []

if 'follow_up_state' in st.session_state:
    st.session_state['follow_up_state'] = False

if 'last_products' in st.session_state:
    st.session_state['last_products'] = None

if 'current_context' in st.session_state:
    st.session_state['current_context'] = None

if 'product_type' in st.session_state:
    st.session_state['product_type'] = None

if 'thinking' in st.session_state:
    st.session_state['thinking'] = False

# Clear any containers that might have content
if 'response_container' in st.session_state:
    st.session_state.response_container.empty()

if 'follow_up_container' in st.session_state:
    st.session_state.follow_up_container.empty()

if 'thinking_container' in st.session_state:
    st.session_state.thinking_container.empty()

print("Session state cleared successfully!") 