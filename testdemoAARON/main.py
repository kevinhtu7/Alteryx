# main.py

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
    def __init__(self):
        load_dotenv()
        self.chroma_client, self.collection = self.initialize_chromadb()
        self.setup_language_model()
        self.setup_langchain()
        self.initialize_tools()

    def initialize_chromadb(self):
        # Initialize ChromaDB client using environment variable for path
        client = db.PersistentClient(path="testdemo/chroma.db")
        collection = client.get_collection(name="Company_Documents")
        return client, collection

    def setup_language_model(self):
        self.repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceHub(
            repo_id=self.repo_id,
            model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50},
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

    def get_context_from_collection(self, input, access_role):
        # Extract context from the collection
        if access_role == "General Access":
            documents = self.collection.query(
                query_texts=[input],
                n_results=5
            )['documents']
        else:
            documents = self.collection.query(
                query_texts=[input],
                n_results=10
            )['documents']
        context = " ".join([doc['content'] for doc in documents])
        return context

    def generate_response(self, input_dict):
        input_dict["question"] = self.preprocess_input(input_dict["question"])
        template = PromptTemplate(input_variables=["context", "question"], template="{context}\nQuestion: {question}\nAnswer:")
        prompt = template.format(**input_dict)
        response = self.llm(prompt)
        return response

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

    def preprocess_input(self, input):
        # Anonymize, spellcheck, and ensure niceness
        anonymized = self.anonymize_text(input)
        spellchecked = self.spellcheck_text(anonymized)
        nice_input = self.ensure_niceness(spellchecked)
        return nice_input
