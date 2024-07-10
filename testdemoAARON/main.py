__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
from dotenv import load_dotenv
import chromadb as db
from chromadb import Client
from chromadb.config import Settings
from langchain_community.llms import HuggingFaceHub, OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import logging
import sqlite3

# Import necessary libraries for anonymization, spellchecking, and ensuring niceness
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from spellchecker import SpellChecker
from textblob import TextBlob
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        if self.llm_option == "Local (PHI3)":
            model_name_or_path = "local_models/phi3_instruct"
            if not os.path.exists(model_name_or_path):
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model_repo_id = "microsoft/Phi-3-mini-4k-instruct"
                self.tokenizer = AutoTokenizer.from_pretrained(model_repo_id)
                self.model = AutoModelForCausalLM.from_pretrained(model_repo_id)
                self.tokenizer.save_pretrained(model_name_or_path)
                self.model.save_pretrained(model_name_or_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        elif self.llm_option == "External (OpenAI)" and self.openai_api_key:
            self.llm = OpenAI(api_key=self.openai_api_key)

    def generate_response(self, prompt):
        if self.llm_option == "Local (PHI3)":
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs)
            response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            return response
        elif self.llm_option == "External (OpenAI)":
            return self.llm(prompt)

    def get_context_from_collection(self, input, access_role):
        # Extract context from the collection
        if access_role == "General Access":
            documents = self.collection.query(query_texts=[input], n_results=5)
        else:
            documents = self.collection.query(query_texts=[input], n_results=10)
        for document in documents["documents"]:
            context = document
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
        context = input_dict.get("context", "")
        question = input_dict.get("question", "")
        combined_text = f"{context} {question}"
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
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt
            | self.generate_response
            | AnswerOnlyOutputParser()
        )

if __name__ == "__main__":
    bot = ChatBot()
