from main import ChatBot
import streamlit as st

# Initialize session state for LLM selection
if 'llm_option' not in st.session_state:
    st.session_state.llm_option = "Local (PHI3)"
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = None
if 'bot' not in st.session_state:
    st.session_state.bot = ChatBot(llm_option=st.session_state.llm_option, openai_api_key=st.session_state.openai_api_key)

st.set_page_config(page_title="Meeting Information Bot")
with st.sidebar:
    st.title('Meeting Information Bot')
    st.session_state.llm_option = st.selectbox(
        "Choose LLM",
        ["Local (PHI3)", "External (OpenAI)"]
    )

    if st.session_state.llm_option == "External (OpenAI)":
        st.session_state.openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
    else:
        st.session_state.openai_api_key = None

    # Re-initialize bot if LLM option changes
    if st.session_state.llm_option != st.session_state.bot.llm_option or \
       (st.session_state.llm_option == "External (OpenAI)" and not st.session_state.openai_api_key):
        st.session_state.bot.unload_language_model()
        if st.session_state.llm_option == "External (OpenAI)" and not st.session_state.openai_api_key:
            st.error("Please enter the OpenAI API key to use the External (OpenAI) model.")
        else:
            st.session_state.bot = ChatBot(llm_option=st.session_state.llm_option, openai_api_key=st.session_state.openai_api_key)

role = st.radio(
    "What's your role",
    ["General Access", "Executive Access"],
    format_func=lambda x: "Executive Access" if x == "Executive Access" else "General Access"
)

# Initialize or maintain the list of past interactions and contexts
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, what can I help you with?"}]
    st.session_state.context_history = []

# Function for generating LLM response
def generate_response(input_dict):
    nice_input = st.session_state.bot.preprocess_input(input_dict)
    result = st.session_state.bot.rag_chain.invoke(nice_input)
    return result

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

    # Retrieve context from the database
    context = st.session_state.bot.get_context_from_collection(input, access_role=role)
    st.session_state.context_history.append(context)  # Store the context for potential future references

    # Generate a new response
    input_dict = {"context": context, "question": input}
    with st.chat_message("assistant"):
        with st.spinner("Grabbing your answer from database..."):
            response = generate_response(input_dict)
            st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
