from dotenv import load_dotenv
import os
import sqlite3
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from chromadb import PersistentClient
from rerankers import Reranker

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
        # Initialize ChromaDB client
        client = PersistentClient()
        self.db_path = os.path.abspath("testdemoFINAL/chroma.db")  # Save the path to be used in other methods

        # Check if the collection exists, if not create it
        try:
            collection = client.get_collection(name="Company_Documents")
        except Exception as e:
            collection = client.create_collection(name="Company_Documents")

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
        all_documents = self.collection.query(query_texts=[input], n_results=10)
        if not all_documents or 'documents' not in all_documents or not all_documents['documents']:
            return "I do not know..."

        filtered_documents = []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for doc in all_documents['documents']:
            doc_id = doc['id']
            cursor.execute("SELECT string_value FROM embedding_metadata WHERE key='access_role' AND id=?", (doc_id,))
            access_role = cursor.fetchone()
            if access_role and access_role[0] in access_levels:
                filtered_documents.append(doc)

        cursor.close()
        conn.close()

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
        If you don't know the answer, simply state "I do not know...".
        If the user does not have access to the required information, state "YOU SHALL NOT PASS!".

        Context: {context}
        Question: {question}
        Answer:
        """

        self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        self.rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | AnswerOnlyOutputParser()
        )

    def generate_response(self, input_dict, access_levels):
        context = self.get_context_from_collection(input_dict['question'], access_levels)
        if context in ["YOU SHALL NOT PASS!", "I do not know..."]:
            return context

        try:
            nice_input = self.preprocess_input(input_dict)
            result = self.rag_chain.invoke(input_dict)
            return result
        except Exception as e:
            return "An error occurred while generating the response."