import streamlit as st
import pandas as pd
import sqlite3
from sqlite3 import Error
import os
from dotenv import load_dotenv

# Set page configuration at the top of the script
st.set_page_config(page_title="Meeting Information Bot")

# Load environment variables (if needed)
load_dotenv()

def create_connection():
    connection = None
    try:
        # Adjust the path to chroma.db based on its location in the repository
        db_path = os.path.join(os.path.dirname(__file__), 'testdemoAARON', 'chroma.db')
        connection = sqlite3.connect(db_path)
    except Error as e:
        st.error(f"Error: '{e}'")
    return connection

def query_database(query, params=None):
    connection = create_connection()
    cursor = connection.cursor()
    try:
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(result, columns=columns)
    except Error as e:
        st.error(f"Error: '{e}'")
        return None
    finally:
        cursor.close()
        connection.close()

def get_user_role(username, password):
    query = "SELECT role FROM Users WHERE UserID = ? AND PW = ?"
    params = (username, password)
    df = query_database(query, params)
    if not df.empty:
        return df.iloc[0]['role']
    else:
        return None

def get_access_level(role):
    query = "SELECT access_levels FROM Roles WHERE role = ?"
    params = (role,)
    df = query_database(query, params)
    if not df.empty:
        return df.iloc[0]['access_levels']
    else:
        return None

def main():
    st.title("Meeting Information Bot")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.access_level = ""

    if not st.session_state.logged_in:
        # Login Page
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            role = get_user_role(username, password)
            if role:
                access_level = get_access_level(role)
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.access_level = access_level
                st.success(f"Welcome {username}! Your role is {role} with {access_level} access.")
                run_app(access_level)
            else:
                st.error("Invalid username or password")
    else:
        run_app(st.session_state.access_level)

def run_app(access_level):
    # Sidebar elements
    with st.sidebar:
        st.title('Meeting Information Bot')

        llm_selection = st.selectbox(
            "Select LLM",
            ["Local (PHI3)", "External (OpenAI)"],
            key="llm_selection"
        )

        if llm_selection == "External (OpenAI)":
            st.session_state.api_key = st.text_input("Enter OpenAI API Key", type="password")
        else:
            st.session_state.api_key = ""

    # Prevent the user from asking questions if OpenAI is selected and no API key is entered
    if st.session_state.llm_selection == "External (OpenAI)" and not st.session_state.api_key:
        st.warning("Please enter your OpenAI API key to proceed.")
    else:
        # Initialize ChatBot with the selected LLM
        try:
            bot = ChatBot(llm_type=st.session_state.llm_selection, api_key=st.session_state.api_key)
        except Exception as e:
            st.error(f"Failed to initialize the chatbot: {e}")
            st.stop()

        # Initialize or maintain the list of past interactions and contexts
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Welcome, what can I help you with?"}]
            st.session_state.context_history = []

        # Function for generating LLM response
        def generate_response(input_dict):
            try:
                nice_input = bot.preprocess_input(input_dict)
                result = bot.rag_chain.invoke(nice_input)
                return result
            except Exception as e:
                st.error(f"Error generating response: {e}")
                return "An error occurred while generating the response."

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
            try:
                context = "Default context for access level: " + access_level  # Placeholder for actual context retrieval
                st.session_state.context_history.append(context)  # Store the context for potential future references
            except Exception as e:
                st.error(f"Error retrieving context: {e}")
                context = "An error occurred while retrieving context."

            # Generate a new response
            input_dict = {"context": context, "question": input}
            with st.chat_message("assistant"):
                with st.spinner("Grabbing your answer from database..."):
                    response = generate_response(input_dict)
                    st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)

if __name__ == '__main__':
    main()
