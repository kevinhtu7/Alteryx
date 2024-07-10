import streamlit as st
from main import ChatBot

st.set_page_config(page_title="Meeting Information Bot")
with st.sidebar:
    st.title('Meeting Information Bot')

role = st.radio(
    "What's your role",
    ["General Access", "Executive Access"],
    format_func=lambda x: "Executive Access" if x == "Executive Access" else "General Access"
)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, what can I help you with?"}]
    st.session_state.context_history = []

# Function for generating LLM response
def generate_response(input_dict):
    nice_input = bot.preprocess_input(input_dict)
    result = bot.rag_chain.invoke(nice_input)
    return result

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Add a selection for the LLM next to the chat input
col1, col2 = st.columns([4, 1])
with col1:
    input = st.chat_input(placeholder="Type your question here...")

with col2:
    llm_option = st.selectbox("Select LLM", ["Local (PHI3)", "External (OpenAI)"], key="llm_selection", index=0)

# If external LLM is selected, ask for OpenAI API key
openai_api_key = None
if llm_option == "External (OpenAI)":
    openai_api_key = st.text_input("Enter OpenAI API Key", type="password")

# Initialize ChatBot after getting the user inputs
bot = ChatBot(llm_option=llm_option, openai_api_key=openai_api_key)

if input:
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

    context = bot.get_context_from_collection(input, access_role=role)
    st.session_state.context_history.append(context)

    input_dict = {"context": context, "question": input}
    with st.chat_message("assistant"):
        with st.spinner("Grabbing your answer from database..."):
            response = generate_response(input_dict)
            st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
