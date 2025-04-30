import streamlit as st

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title("Simple Chat Test")

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

    # Add bot response
    with st.chat_message("assistant"):
        st.write(f"Echo: {prompt}")
        st.session_state.messages.append({"role": "assistant", "content": f"Echo: {prompt}"})

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun() 