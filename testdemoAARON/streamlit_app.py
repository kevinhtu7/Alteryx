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
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Welcome, what can I help you with?"}]
        st.session_state.context_history = []

    def generate_response(input_dict):
        result = bot.rag_chain.invoke(input_dict)
        return result

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if input := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": input})
        with st.chat_message("user"):
            st.write(input)

        context = bot.get_context_from_collection(input, access_role=role)
        st.session_state.context_history.append(context)

        analyzer_results = bot.analyze_text(input)
        anonymized_input = bot.anonymize_text(input, analyzer_results)
        corrected_input = bot.check_spelling(anonymized_input)
        sentiment = bot.analyze_sentiment(corrected_input)

        st.write(f"Sentiment Analysis: {sentiment}")

        input_dict = {"context": context, "question": corrected_input}
        with st.chat_message("assistant"):
            with st.spinner("Grabbing your answer from database..."):
                response = generate_response(input_dict)
                st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
else:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Welcome, what can I help you with?"}]
        st.session_state.context_history = []

    def generate_response(input_dict):
        result = bot.rag_chain.invoke(input_dict)
        return result

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if input := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": input})
        with st.chat_message("user"):
            st.write(input)

        context = bot.get_context_from_collection(input, access_role=role)
        st.session_state.context_history.append(context)

        analyzer_results = bot.analyze_text(input)
        anonymized_input = bot.anonymize_text(input, analyzer_results)
        corrected_input = bot.check_spelling(anonymized_input)
        sentiment = bot.analyze_sentiment(corrected_input)

        st.write(f"Sentiment Analysis: {sentiment}")

        input_dict = {"context": context, "question": corrected_input}
        with st.chat_message("assistant"):
            with st.spinner("Grabbing your answer from database..."):
                response = generate_response(input_dict)
                st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
