from dotenv import load_dotenv
import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb as db
from chromadb import Client
from chromadb.config import Settings
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import logging
import sqlite3

# Import necessary libraries for anonymization, spellchecking, and niceness
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from spellchecker import SpellChecker
from textblob import TextBlob

# Setup logging
logging.basicConfig(level=logging.INFO)

# class AnswerOnlyOutputParser(StrOutputParser):
#     def parse(self, response):
#         # Extract the answer from the response
#         return response.split("Answer:")[1].strip() if "Answer:" in response else response.strip()

class AnswerOnlyOutputParser(StrOutputParser):
    def parse(self, response):
        if "you do not have access" in response.lower():
            return "You do not have access"
        elif "no relevant documents found" in response.lower():
            return "No relevant documents found"
        elif "Answer:" in response:
            return response.split("Answer:")[1].strip()
        else:
            return response.strip()

class ChatBot():
    def __init__(self, llm_type="Local (PHI3)", api_key=""):
        load_dotenv()
        # self.chroma_db_path = os.getenv("CHROMA_DB_PATH")
        # if not self.chroma_db_path:
        #     raise ValueError("CHROMA_DB_PATH environment variable not set or empty.")
        # os.environ["CHROMA_DB_PATH"] = self.chroma_db_path
        
        self.chroma_client, self.collection = self.initialize_chromadb()
        self.llm_type = llm_type
        self.api_key = api_key
        self.setup_language_model()
        self.setup_langchain()
        # Uncomment this line if `initialize_tools` is necessary
        # self.initialize_tools()

    def initialize_chromadb(self):
        # Initialize ChromaDB client using environment variable for path
        db_path = "testdemoAARON/chroma.db"
        client = db.PersistentClient(path=db_path)
        collection = client.get_collection(name="Company_Documents")
        # Verify or create the collection
        # try:
        #     collection = client.get_collection(name="Company_Documents")
        # except Exception as e:
        #     print(f"Error fetching collection: {e}. Creating a new collection.")
        #     client.create_collection(name="Company_Documents", metadata={"description": "Company related documents"})
        #     collection = client.get_collection(name="Company_Documents")

        return client, collection

    def setup_language_model(self):
        if self.llm_type == "External (OpenAI)" and self.api_key:
            try:
                self.repo_id = "openai/gpt-3.5-turbo"  # Update this to the actual repo ID for the external model
                self.llm = HuggingFaceHub(
                    repo_id=self.repo_id,
                    model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50},
                    huggingfacehub_api_token=self.api_key
                )
            except Exception as e:
                raise ValueError(f"Failed to initialize the external LLM: {e}")
        else:
            # Setup for Local (PHI3) model
            try:
                self.repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
                self.llm = HuggingFaceHub(
                    repo_id=self.repo_id,
                    model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50},
                    huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
                )
            except Exception as e:
                raise ValueError(f"Failed to initialize the local LLM: {e}")

    def get_context_from_collection(self, input, access_levels):
        try:
            logging.info(f"Querying all documents for input: {input}")
            # Check if context exists without any role restrictions
            all_documents = self.collection.query(query_texts=[input], n_results=10)
            logging.info(f"All documents: {all_documents}")

            if not all_documents["documents"]:
                logging.info("No relevant documents found for the input.")
                return "No relevant documents found"

            logging.info(f"Querying role-based documents for input: {input} with access levels: {access_levels}")
            # Now check with role-based access levels
            role_documents = []
            if len(access_levels) == 1:
                role_documents = self.collection.query(query_texts=[input], n_results=10, where={"access_role": {"$eq": access_levels[0]["access_role"]}})
            else:
                role_documents = self.collection.query(query_texts=[input], n_results=10, where={"$or": [{"access_role": {"$eq": level["access_role"]}} for level in access_levels]})
            
            logging.info(f"Role-based documents: {role_documents}")

            if not role_documents["documents"]:
                logging.info("User does not have access to the relevant documents.")
                return "You do not have access"

            context = " ".join([doc["text"] for doc in role_documents["documents"]])
            return context
        except Exception as e:
            logging.error(f"Error retrieving context: {e}")
            return "An error occurred while retrieving context."

    # def get_context_from_collection(self, input, access_role):
    #     # Extract context from the collection
    #     if access_role == "General Access":
    #         documents = self.collection.query(
    #             query_texts=[input],
    #             n_results=5
    #         )
    #     else:
    #         documents = self.collection.query(
    #             query_texts=[input],
    #             n_results=10
    #         )
    #     for document in documents["documents"]:
    #         context = document
    #     return context

    # Uncomment this method if it's necessary
    # def initialize_tools(self):
    #     # Initialize tools for anonymization, spellchecking, and ensuring niceness
    #     self.analyzer = AnalyzerEngine()
    #     self.anonymizer = AnonymizerEngine()
    #     self.spellchecker = SpellChecker()

    def preprocess_input(self, input_dict):
        # Anonymize, spellcheck, and ensure niceness
        # Extract context and question from input_dict
        context = input_dict.get("context", "")
        question = input_dict.get("question", "")

        # Concatenate context and question
        combined_text = f"{context} {question}"
        return combined_text

    def setup_langchain(self):
        template = """
        You are an informational chatbot. These employees will ask you questions about company data and meeting information. Use the following piece of context to answer the question.
        If you don't know the answer, just say you don't know. Please provide the file used for context.
        # You answer with short and concise answers, no longer than 2 sentences.

        Context: {context}
        Question: {question}
        Answer:
        """

        self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        self.rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}  # Using passthroughs for context and question
            | self.prompt
            | self.llm
            | AnswerOnlyOutputParser()
        )

