import streamlit as st

st.title("BetterHome Assistant - Reset")

if st.button("Reset Conversation History"):
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
    
    st.success("✅ History has been cleared successfully! You can now return to the main app.")
    st.info("To return to the main app, close this tab and reopen the main application.")

st.markdown("---")
st.markdown("This utility resets the conversation history and clears all temporary data in the BetterHome Assistant.") 