from main import ChatBot
import streamlit as st

# Initialize session state for API key and LLM selection
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "llm_selection" not in st.session_state:
    st.session_state.llm_selection = "Local (PHI3)"

# Sidebar elements
st.set_page_config(page_title="Meeting Information Bot")
with st.sidebar:
    st.title('Meeting Information Bot')
    role = st.radio(
        "What's your role",
        ["General Access", "Executive Access"],
        format_func=lambda x: "Executive Access" if x == "Executive Access" else "General Access"
    )

    llm_selection = st.selectbox(
        "Select LLM",
        ["Local (PHI3)", "External (OpenAI)"],
        key="llm_selection"
    )

    if llm_selection == "External (OpenAI)":
        st.session_state.api_key = st.text_input("Enter OpenAI API Key", type="password")
    else:
        st.session_state.api_key = ""

# Initialize ChatBot with the selected LLM
bot = ChatBot(llm_type=st.session_state.llm_selection, api_key=st.session_state.api_key)

if role == "Executive Access":
    # Initialize or maintain the list of past interactions and contexts
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

    # User-provided prompt
    if input := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": input})
        with st.chat_message("user"):
            st.write(input)

        # Retrieve context from the database
        context = bot.get_context_from_collection(input, access_role=role)
        st.session_state.context_history.append(context)  # Store the context for potential future references

        # Generate a new response
        input_dict = {"context": context, "question": input}
        with st.chat_message("assistant"):
            with st.spinner("Grabbing your answer from database..."):
                response = generate_response(input_dict)
                st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
else:
    # Initialize or maintain the list of past interactions and contexts
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

    # User-provided prompt
    if input := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": input})
        with st.chat_message("user"):
            st.write(input)

        # Retrieve context from the database
        context = bot.get_context_from_collection(input, access_role=role)
        st.session_state.context_history.append(context)  # Store the context for potential future references

        # Generate a new response
        input_dict = {"context": context, "question": input}
        with st.chat_message("assistant"):
            with st.spinner("Grabbing your answer from database..."):
                response = generate_response(input_dict)
                st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)

