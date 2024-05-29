from main import ChatBot
import streamlit as st

bot = ChatBot()

st.set_page_config(page_title="Random Fortune Telling Bot")
with st.sidebar:
    st.title('Random Fortune Telling Bot')

# Function for generating LLM response
def generate_response(input):
    result = bot.get_response(input)
    return result

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, let's unveil your future"}]

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    # Generate a new response if the last message is not from the assistant
    if st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Getting your answer from mystery stuff..."):
                response = generate_response(input)
                st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)
                
    # Append user input to session state without displaying it
    st.session_state.messages.append({"role": "user", "content": input})
    
    # Remove the user's input from session state to prevent it from being displayed
    st.session_state.messages = [msg for msg in st.session_state.messages if msg["role"] != "user"]
