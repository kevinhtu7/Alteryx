#Import dependencies
#!pip install -r requirements.txt
#!pip install textract
#!pip install PyMuPDF
#!pip install python-docx


import os
from dotenv import load_dotenv
import textract
import fitz  # PyMuPDF pip install
import docx
import pandas as pd
from typing import List
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import pinecone
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFaceHub
import streamlit as st

class ChatBot():
  def __init__(self):
    #Implement document loader that can handle text, PDFs, and other formats
    load_dotenv()
    self.documents = [] 
    self.load_documents() 
    self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
    self.docs = self.text_splitter.split_documents(self.documents)
    self.embeddings = HuggingFaceEmbeddings()


    pinecone.init(
          api_key= os.getenv('PINECONE_API_KEY'),
         environment='gcp-starter'
    )

    self.index_name = "langchain-demo"

    if self.index_name not in pinecone.list_indexes():
        pinecone.create_index(name=self.index_name, metric="cosine", dimension=768)
        self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, index_name=self.index_name)
    else:
        self.docsearch = Pinecone.from_existing_index(self.index_name, self.embeddings)

    self.repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    self.llm = HuggingFaceHub(
        repo_id=self.repo_id, model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50}, huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
    )

    self.template = """
    You are a helpful support agent. Employees will ask you questions pertaining to the provided company data and policies. Use the following pieces of context to answer the question. 
     If you don't know the answer, just say you don't know but offer the company documents that are close enough vector-wise but don't qutie fit within the k-nearest neighbors threshhold. 
    You answer with short and concise answer, no longer than 10 sentences and containing as many quotes as possible from the company documents.

     Context: {context}
    Question: {question}
    Answer: 

    """

    self.prompt = PromptTemplate(template=self.template, input_variables=["context", "question"])

    self.rag_chain = (
        {"context": self.docsearch.as_retriever(),  "question": RunnablePassthrough()} 
        | self.prompt 
        | self.llm
        | StrOutputParser() 
    )

def load_documents(self):
       # Load documents from streamlit file uploader (docx, pdf, txt, xlsx, csv, images for OCR) 
    uploaded_file = st.file_uploader("Choose a file", type=["docx", "pdf", "txt", "xlsx", "csv", "jpg", "png"])
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            text = ""
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf:
                for page in pdf:
                    text += page.get_text()
            self.documents.append(text)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            text = ""
            for para in doc.paragraphs:
                text += para.text
            self.documents.append(text)
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
            self.documents.append(text)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)
            text = ""
            for col in df.columns:
                text += " ".join(df[col].astype(str).values)
            self.documents.append(text)
        elif uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            text = ""
            for col in df.columns:
                text += " ".join(df[col].astype(str).values)
            self.documents.append(text)
        elif uploaded_file.type == "image/jpeg" or uploaded_file.type == "image/png":
            text = textract.process(uploaded_file)
            self.documents.append(text.decode("utf-8"))

  
  