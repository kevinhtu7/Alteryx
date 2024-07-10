__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
from dotenv import load_dotenv
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

class AnswerOnlyOutputParser(StrOutputParser):
    def parse(self, response):
        # Extract the answer from the response
        return response.split("Answer:")[1].strip() if "Answer:" in response else response.strip()

class ChatBot():
    def __init__(self, llm_option="Local", openai_api_key=""):
        load_dotenv()
        self.llm_option = llm_option
        self.openai_api_key = openai_api_key
        self.chroma_client, self.collection = self.initialize_chromadb()
        self.setup_language_model()
        self.setup_langchain()
        self.initialize_tools()

    def initialize_chromadb(self):
        # Initialize ChromaDB client using environment variable for path
        client = db.PersistentClient(path="testdemoAARON/chroma.db")
        collection = client.get_collection(name="Company_Documents")
        return client, collection

    def setup_language_model(self):
        if self.llm_option == "OpenAI" and self.openai_api_key:
            from langchain_community.llms import OpenAI
            self.llm = OpenAI(api_key=self.openai_api_key)
        else:
            self.repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
            self.llm = HuggingFaceHub(
                repo_id=self.repo_id,
                model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50},
                huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
            )

    def get_context_from_collection(self, input_text, access_role):
        # Extract context from the collection
        if access_role == "General Access":
            documents = self.collection.query(
                query_texts=[input_text],
                n_results=5
            )
        else:
            documents = self.collection.query(
                query_texts=[input_text],
                n_results=10
            )

        cleaned_documents = []
        for doc in documents["documents"]:
            if isinstance(doc, list):
                doc = ' '.join(doc)
            if 'URL' not in doc:
                cleaned_documents.append(doc)

        context = " ".join(cleaned_documents)
        return context

    def initialize_tools(self):
        # Initialize tools for anonymization, spellchecking, and ensuring niceness
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.spellchecker = SpellChecker()

    def anonymize_text(self, text):
        analyzer_results = self.analyzer.analyze(text=text, language="en")
        anonymized_text = self.anonymizer.anonymize(text=text, analyzer_results=analyzer_results).text
        return anonymized_text

    def spellcheck_text(self, text):
        corrected_text = self.spellchecker.correction(text)
        return corrected_text

    def ensure_niceness(self, text):
        blob = TextBlob(text)
        nice_text = " ".join(blob.words)
        return nice_text

    def preprocess_input(self, input_dict):
        # Anonymize, spellcheck, and ensure niceness
        # Extract context and question from input_dict
        context = input_dict.get("context", "")
        question = input_dict.get("question", "")
        
        # Concatenate context and question
        combined_text = f"{context} {question}"
        
        # Anonymize, spellcheck, and ensure niceness
        anonymized = self.anonymize_text(combined_text)
        spellchecked = self.spellcheck_text(anonymized)
        nice_input = self.ensure_niceness(spellchecked)
        return nice_input

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

