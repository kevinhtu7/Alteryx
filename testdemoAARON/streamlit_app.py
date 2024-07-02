# streamlit_app.py
from main import ChatBot
import streamlit as st

bot = ChatBot()

st.set_page_config(page_title="Meeting Information Bot")
with st.sidebar:
    st.title('Meeting Information Bot')

role = st.radio(
    "What's your role",
    ["General Access", "Executive Access"],
    format_func=lambda x: "Executive Access" if x == "Executive Access" else "General Access"
)

# Store and display chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, what can I help you with?"}]
    st.session_state.context_history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

    # Debugging output
    st.write("Input received:", input)

    # Retrieve context from the database
    context = bot.get_context_from_collection(input, access_role=role)
    st.write("Context retrieved:", context)
    st.session_state.context_history.append(context)

    # Generate a new response
    input_dict = {"context": context, "question": input}
    with st.chat_message("assistant"):
        with st.spinner("Grabbing your answer from database..."):
            response = bot.generate_response(input_dict)
            st.write("Response generated:", response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
