from main import ChatBot
import streamlit as st

bot = ChatBot()
    
st.set_page_config(page_title="Meeting Information Bot")
with st.sidebar:
    st.title('Meeting Information Bot')



role = st.radio(
    "What's your role",
    ["Admin", "General"],
    captions = ["All access", "Restricted Access"])

if role == "Admin":
    

    # Function for generating LLM response
    def generate_response(input):
        result = bot.rag_chain.invoke(input)
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

    # Generate a new response if the last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Grabbing your answer from database..."):
                response = generate_response(input)
                st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
            
else st.write("You are not an admin")
