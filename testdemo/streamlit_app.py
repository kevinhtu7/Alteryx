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

if role == "Executive Access":
    # Initialize or maintain the list of past interactions and contexts
    if "messages" not in st.session_state: #
        st.session_state.messages = [{"role": "assistant", "content": "Welcome, what can I help you with?"}] #
        st.session_state.context_history = [] #
        
    # Function for generating LLM response
    def generate_response(input_dict):
        result = bot.generate_response(input_dict)
        return result

    # Store LLM generated responses
    # if "messages" not in st.session_state.keys():
        # st.session_state.messages = [{"role": "assistant", "content": "Welcome, what can I help you with?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if input := st.chat_input():
        #spellcheck 
        corrected_input = bot.spellcheck(input)
        st.session_state.messages.append({"role": "user", "content": input})
        with st.chat_message("user"):
            st.write(input)

        # Retrieve context from the database
        context = bot.get_context_from_collection(input, access_role=role) #
        st.session_state.context_history.append(context)  # Store the context for potential future references

        # Generate a new response
        # context = bot.get_context_from_collection(input, access_role=role)
        input_dict = {"context": context, "question": input}
        with st.chat_message("assistant"):
            with st.spinner("Grabbing your answer from database..."):
                response = generate_response(input_dict)
                st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
else:
    # Initialize or maintain the list of past interactions and contexts
    if "messages" not in st.session_state: #
        st.session_state.messages = [{"role": "assistant", "content": "Welcome, what can I help you with?"}] #
        st.session_state.context_history = [] #
        
     # Function for generating LLM response
    def generate_response(input_dict):
        result = bot.rag_chain.invoke(input_dict)
        return result

    # Store LLM generated responses
    # if "messages" not in st.session_state.keys():
        # st.session_state.messages = [{"role": "assistant", "content": "Welcome, what can I help you with?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if input := st.chat_input():
        #spellcheck layer 
        corrected_input = bot.spellcheck(input)
        st.session_state.messages.append({"role": "user", "content": input})
        with st.chat_message("user"):
            st.write(input)

        # Retrieve context from the database
        context = bot.get_context_from_collection(input, access_role=role) #
        st.session_state.context_history.append(context)  # Store the context for potential future references
        
        # Generate a new response
        # context = bot.get_context_from_collection(input, access_role=role)
        input_dict = {"context": context, "question": input}
        with st.chat_message("assistant"):
            with st.spinner("Grabbing your answer from database..."):
                response = generate_response(input_dict)
                st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
