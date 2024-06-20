import os
from dotenv import load_dotenv
from chromadb import Client
from chromadb.config import Settings
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


class ChatBot():
  def __init__(self):
      load_dotenv()
      self.chroma_client, self.collection = self.initialize_chromadb()
      self.setup_language_model()
      self.setup_langchain()

  def initialize_chromadb(self):
    try:
        # Assuming ChromaDB can utilize SQLite directly, which is typically not standard
        client = Client(Settings(
            chroma_db_impl="sqlite",
            db_path="./chroma.sqlite3"  # Path to your SQLite file
        ))
        collection = client.get_or_create_collection(name="Company_Documents")
        print("Initialized ChromaDB instance with SQLite.")
    except ValueError as e:
        print("Using existing ChromaDB instance:", e)
        client = Client()
        collection = client.get_or_create_collection(name="Company_Documents")
    return client, collection
  
  def setup_language_model(self):
      self.repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
      self.llm = HuggingFaceHub(
          repo_id=self.repo_id,
          model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50},
          huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
      )

  def setup_langchain(self):
      template = """
      You are an informational chatbot. These employees will ask you questions about company data and meeting information. Use the following piece of context to answer the question.
      If you don't know the answer, just say you don't know.
      You answer with short and concise answers, no longer than 2 sentences.

      Context: {context}
      Question: {question}
      Answer:
      """

      self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])
      self.rag_chain = (
          {"context": self.collection, "question": RunnablePassthrough()}  # Adjusted for direct use of the ChromaDB collection
          | self.prompt
          | self.llm
          | AnswerOnlyOutputParser()
      )

class AnswerOnlyOutputParser(StrOutputParser):
    def parse(self, response):
          # Extract the answer from the response
        return response.split("Answer:")[1].strip() if "Answer:" in response else response.strip()
