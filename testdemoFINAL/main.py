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
from rerankers import Reranker
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from spellchecker import SpellChecker
from textblob import TextBlob

class AnswerOnlyOutputParser(StrOutputParser):
    def parse(self, response):
        if "you do not have access" in response.lower():
            return "YOU SHALL NOT PASS!"
        return response.split("Answer:")[1].strip() if "Answer:" in response else response.strip()

class ChatBot():
    def __init__(self, llm_type="Local (PHI3)", api_key=""):
        load_dotenv()
        self.chroma_client, self.collection = self.initialize_chromadb()
        self.llm_type = llm_type
        self.api_key = api_key
        self.setup_language_model()
        self.setup_langchain()
        self.setup_reranker()

    def setup_reranker(self):
        self.reranker = Reranker("t5")

    def rerank_documents(self, question, documents):
        context = " ".join([doc["text"] for doc in documents])
        reranked_documents = self.reranker.rank(question, context)
        return reranked_documents

    def initialize_chromadb(self):
        # Initialize ChromaDB client using environment variable for path
        db_path = os.path.abspath("testdemoFINAL/chroma.db")
        print(f"Database path: {db_path}")  # Log the path to ensure it's correct
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found at {db_path}")
        try:
            client = db.PersistentClient(path=db_path)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to the database at {db_path}: {e}")
        collection = client.get_collection(name="Company_Documents")
        self.db_path = db_path  # Save the path to be used in other methods
        return client, collection

    def setup_language_model(self):
        if self.llm_type == "External (OpenAI)" and self.api_key:
            try:
                self.repo_id = "openai/gpt-4o-mini"
                self.llm = HuggingFaceHub(
                    repo_id=self.repo_id,
                    model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50},
                    huggingfacehub_api_token=self.api_key
                )
            except Exception as e:
                raise ValueError(f"Failed to initialize the external LLM: {e}")
        else:
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
        # Fetch all documents
        all_documents = self.collection.query(query_texts=[input], n_results=100)
        if not all_documents or 'documents' not in all_documents or not all_documents['documents']:
            return "I do not know..."

        print(f"All documents retrieved: {all_documents}")

        filtered_documents = []

        # Filter documents based on access levels in metadata
        for doc in all_documents['documents']:
            print(f"Processing document: {doc}")
            if isinstance(doc, dict):
                metadata = doc.get('embedding_metadata', [])
                print(f"Metadata: {metadata}")
                if isinstance(metadata, list):
                    access_roles = [item['string_value'] for item in metadata if item.get('key') == 'access_role']
                    print(f"Access roles for document: {access_roles}")
                    if any(role in access_levels for role in access_roles):
                        filtered_documents.append(doc)
                else:
                    print(f"Metadata is not a list: {metadata}")
            else:
                print(f"Document is not a dictionary: {doc}")

        if not filtered_documents:
            return "YOU SHALL NOT PASS!"

        reranked_documents = self.rerank_documents(input, filtered_documents)
        context = " ".join([doc["text"] for doc in reranked_documents[:3]])
        return context

    def preprocess_input(self, input_dict):
        context = input_dict.get("context", "")
        question = input_dict.get("question", "")
        combined_text = f"{context} {question}"
        return combined_text

    def setup_langchain(self):
        template = """
        You are an informational chatbot. These employees will ask you questions about company data and meeting information. Use the following piece of context to answer the question.
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
