from main import ChatBot
import streamlit as st

bot = ChatBot()

st.set_page_config(page_title="Meeting Information Bot")
with st.sidebar:
    st.title('Meeting Information Bot')

role = st.radio(
    "What's your role",
    ["Admin", "General"],
    format_func=lambda x: "All access" if x == "Admin" else "Restricted Access"
)

if role == "Admin":
    # Function for generating LLM response
    def generate_response(input_dict):
        result = bot.rag_chain.invoke(input_dict)
        return result

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Welcome, what can I help you with?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if input := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": input})
        with st.chat_message("user"):
            st.write(input)

        # Generate a new response
        context = bot.get_context_from_collection()
        input_dict = {"context": context, "question": input}
        with st.chat_message("assistant"):
            with st.spinner("Grabbing your answer from database..."):
                response = generate_response(input_dict)
                st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
else:
    st.write("You are not an admin")
