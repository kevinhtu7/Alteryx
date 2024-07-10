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
import openai
import logging
import sqlite3

# Import necessary libraries for anonymization, spellchecking, and niceness
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from spellchecker import SpellChecker
from textblob import TextBlob
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class AnswerOnlyOutputParser(StrOutputParser):
    def parse(self, response):
        # Extract the answer from the response
        return response.split("Answer:")[1].strip() if "Answer:" in response else response.strip()

class ChatBot():
    def __init__(self, llm_option="Local (PHI3)", openai_api_key=None):
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
        if self.llm_option == "External (OpenAI)" and self.openai_api_key:
            openai.api_key = self.openai_api_key
            self.llm = openai.ChatCompletion.create
        else:
            self.repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
            self.llm = HuggingFaceHub(
                repo_id=self.repo_id,
                model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50},
                huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
            )

    def get_context_from_collection(self, input, access_role):
        # Extract context from the collection
        n_results = 5 if access_role == "General Access" else 10
        documents = self.collection.query(
            query_texts=[input],
            n_results=n_results
        )
        
        context = " ".join([" ".join(doc) if isinstance(doc, list) else doc for doc in documents["documents"]])
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

    def generate_response(self, input_dict):
        nice_input = self.preprocess_input(input_dict)
        if self.llm_option == "External (OpenAI)" and self.openai_api_key:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.prompt.template.format(context=nice_input['context'], question=nice_input['question'])},
                    {"role": "user", "content": nice_input['question']}
                ]
            )
            return response.choices[0].message['content']
        else:
            result = self.rag_chain.invoke(nice_input)
            return result

    def unload_language_model(self):
        self.llm = None
