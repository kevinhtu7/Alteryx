__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
from dotenv import load_dotenv
import chromadb as db
from chromadb import Client
from chromadb.config import Settings
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import logging
import sqlite3

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

    # def get_context_from_collection(self, input, access_role):
    #     # Extract context from the collection
    #     if access_role == "General Access":
    #         documents = self.collection.query(query_texts=[input],
    #                                           n_results=3,
    #                                           where={"access_role": access_role}
    #                                             )
    #     # elif access_role == "Executive Access":
    #     #     documents = self.collection.query(query_texts=[input],
    #     #                                       n_results=3
    #     #                                         )

    #     elif access_role == "Executive Access":
    #         # access_text = "["+"{"+ "access_role: "+ "General Access"+ "}" + "," +  "{"+"access_role: " + access_role +"]"   NOT WORKING
    #         documents = self.collection.query(query_texts=[input],
    #                                           n_results=3,
    #                                           where={
    #                                           # "$or": [{"access_role": "General Access"}, {"access_role": access_role}   WORKS WITH ORIGINAL CODE
    #                                             "$or": access_text
    #                                           }
    #                                             )


    def get_context_from_collection(self, input, access_role):
    # Extract context from the collection
        if access_role == "General Access":
            documents = self.collection.query(query_texts=[input],
                                          n_results=3,
                                          where={"access_role": access_role}
                                          )
        elif access_role == "Executive Access":
            access_text = [{"access_role": "General Access"}, {"access_role": access_role}]
            documents = self.collection.query(query_texts=[input],
                                          n_results=3,
                                          where={"$or": access_text}
                                          )
        
        
        for document in documents["documents"]:
            context = document
        return context

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
