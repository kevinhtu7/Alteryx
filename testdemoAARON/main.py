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
#from transformers import T5ForConditionalGeneration, T5Tokenizer
from rerankers import Reranker
#from py2neo import Graph

# Import necessary libraries for anonymization, spellchecking, and niceness
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from spellchecker import SpellChecker
from textblob import TextBlob

#class AnswerOnlyOutputParser(StrOutputParser):
#    def parse(self, response):
        # Extract the answer from the response
 #       return response.split("Answer:")[1].strip() if "Answer:" in response else response.strip()

class AnswerOnlyOutputParser(StrOutputParser):
    def parse(self, response):
        if "you do not have access" in response.lower():
            return "You do not have access"
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
        #self.initialize_knowledge_graph()
        # Uncomment this line if `initialize_tools` is necessary
        # self.initialize_tools()

    def setup_reranker(self):
        self.reranker = Reranker("t5")

    def rerank_documents(self, question, documents):
        # Get the context from the collection
        for document in documents["documents"]:
            context = document
        # Rerank the documents
        reranked_documents = self.reranker.rank(question, context)
        return reranked_documents    
        
    def initialize_chromadb(self):
        # Initialize ChromaDB client using environment variable for path
        db_path = "testdemoAARON/chroma.db"
        client = db.PersistentClient(path=db_path)
        collection = client.get_collection(name="Company_Documents")
        return client, collection

    def setup_language_model(self):
        if self.llm_type == "External (OpenAI)" and self.api_key:
            try:
                self.repo_id = "openai/gpt-4o-mini"  # Update this to the actual repo ID for the external model
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

    #def initialize_knowledge_graph(self):
        #neo4j_url = os.getenv('NEO4J_URL')
        #neo4j_user = os.getenv('NEO4J_USER')
        #neo4j_password = os.getenv('NEO4J_PASSWORD')
        #self.graph = Graph(neo4j_url, auth=(neo4j_user, neo4j_password))

    #def query_knowledge_graph(self, query):
        #return self.graph.run(query).data()
    
    def get_context_from_collection(self, input, access_levels):
        # Query all context first
        all_documents = self.collection.query(query_texts=[input], n_results=100)

        if not all_documents or 'documents' or not all_documents.get('documents'):
            return "No context found for the given input."

        all_documents = all_documents['documents']

        print(f'All documents: {all_documents}')

        # access_level check 
        if len(access_levels) == 1:
            where_clause = {"access_role": access_levels[0]}
        else:
            where_clause = {"$or": [{"access_role": level} for level in access_levels]}

        print(f'Where clause: {where_clause}')

        documents = self.collection.query(
            query_texts=[input], 
            n_results=100, 
            where=where_clause
        )   

        if not documents or 'documents' or not documents.get('documents'):
            return "No context available for your access level."
        
        documents = documents['documents']

        print(f"Filtered documents: {documents}")

        # Rerank the filtered documents
        reranked_documents = self.rerank_documents(input, documents)

        # Use top 3 reranked documents
        context = " ".join([doc["text"] for doc in reranked_documents[:3]])  # Append the top 3 docs together
        # context = reranked_documents[0]["text"]  # Pick the best document from the top 3

        return context
            

    # def get_context_from_collection(self, input, access_levels):
    #     # Extract context from the collection
    #     if len(access_levels) == 1:
    #         documents = self.collection.query(query_texts=[input],
    #                                       n_results=10,
    #                                       #where={"access_role": "General Access"}
    #                                       where=access_levels[0]
    #                                       )
    #     # if access_role == "General":
    #    #      documents = self.collection.query(query_texts=[input],
    #    #                                   n_results=5,
    #    #                                   where={"access_role": access_role+" Access"}
    #    #                                   )
    #    # elif access_role == "Executive":
    #    #     access_text = [{"access_role": "General Access"}, {"access_role": "Executive Access"}]
    #    #     documents = self.collection.query(query_texts=[input],
    #    #                                   n_results=10,
    #    #                                   where={"$or": access_text}
    #    #                                   )
    #     else:
    #         documents = self.collection.query(query_texts=[input],
    #                                           n_results=10,
    #                                           where={"$or": access_levels}
    #                                           )
    #     reranked_documents = self.rerank_documents(input, documents)
    #     # Use top 3 reranked documents
    #     context = " ".join([doc.text for doc in reranked_documents.top_k(3)])  # This code is append the top 3 docs together
    #     # context = reranked_documents.top_k(3)[0].text # This code is to pick the best document from the top 3
    #     return context

    #def get_context_from_knowledge_graph(self, input):
        # query for everything
        #query = "MATCH (n)-[r]->(m) RETURN n, r, m"
        #query = f"MATCH (n) WHERE n.name CONTAINS '{input}' RETURN n"
        #results = self.query_knowledge_graph(query)
        #results = ["", ""]
        #context = " ".join([str(result) for result in results])
        #return context
        #for document in documents["documents"]:
           #context = document
        #reranked_documents = self.rerank_documents(input, documents["documents"])
        #context = " ".join([doc["text"] for doc in reranked_documents[:5]])  # Use top 5 reranked documents
        #context = reranked_documents  # Use top 5 reranked documents
        #return context 


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
        If you don't know the answer, simply state "You do not have the required level of access".
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

    #def get_combined_context(self, input, access_levels):
        #collection_context = self.get_context_from_collection(input, access_levels)
        #graph_context = self.get_context_from_knowledge_graph(input)
        #combined_context = f"{collection_context} {graph_context}"
        #return combined_context

    #def answer_question(self, input_dict, access_levels):
        ## input_text = self.preprocess_input(input_dict)
        #combined_context = self.get_combined_context(input_dict, access_levels)
        #response = self.rag_chain.run({"context": combined_context, "question": input_dict.get("question", "")})
        #return response
