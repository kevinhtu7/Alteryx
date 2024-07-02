from main import ChatBot
import streamlit as st

print("hello2")

# Initialize the ChatBot
st.write("Initializing ChatBot...")
bot = ChatBot()
st.write("ChatBot initialized.")

# Set the page title
st.set_page_config(page_title="Meeting Information Bot")

# Sidebar for role selection
with st.sidebar:
    st.title('Meeting Information Bot')
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
    result = bot.rag_chain.invoke(input_dict)
    return result

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
input = st.chat_input("Type your message here...")

if input:
    st.write("User Input:", input)
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

    # Retrieve context from the database
    context = bot.get_context_from_collection(input, access_role=role)
    st.write("Context:", context)
    st.session_state.context_history.append(context)

    # Analyze, anonymize, and correct the input text
    analyzer_results = bot.analyze_text(input)
    anonymized_input = bot.anonymize_text(input, analyzer_results)
    corrected_input = bot.check_spelling(anonymized_input)
    sentiment = bot.analyze_sentiment(corrected_input)
    st.write("Sentiment Analysis:", sentiment)

    # Generate a new response
    input_dict = {"context": context, "question": corrected_input}
    with st.chat_message("assistant"):
        with st.spinner("Grabbing your answer from database..."):
            response = generate_response(input_dict)
            st.write(response)
        message = {"role": "assistant", "content": response}
        st.write("Generated Response:", response)
        st.session_state.messages.append(message)
