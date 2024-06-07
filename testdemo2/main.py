# Import dependencies
#!pip install -r requirements.txt
#!pip install textract
#!pip install PyMuPDF
#!pip install python-docx
#!pip install python-pptx
#!pip install pytesseract
#!pip install Pillow
#!pip install streamlit
#!pip install langchain
#!pip install huggingface-hub
#!pip install pinecone-client


import os
from dotenv import load_dotenv
import textract
import fitz  # PyMuPDF
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
from pptx import Presentation
import pytesseract
from PIL import Image

class ChatBot():
    load_dotenv()
    # loader = TextLoader('./horoscope.txt')
    loader = DocumentLoader()
    # documents = loader.load()
    documents = loader.load_documents()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    pinecone.init(
        api_key= os.getenv('PINECONE_API_KEY'),
        environment='gcp-starter'
    )

    index_name = "langchain-demo2"

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, metric="cosine", dimension=768)
        docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    else:
        docsearch = Pinecone.from_existing_index(index_name, embeddings)

    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceHub(
      repo_id=repo_id, model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50}, huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
    )

    from langchain import PromptTemplate

    template = """
        You are a helpful support agent. Employees will ask you questions pertaining to the provided company data and policies. Use the following pieces of context to answer the question. 
        If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer. 
        You answer with short and concise answer, no longer than 10 sentences and containing as many quotes as possible from the company documents.

         Context: {context}
        Question: {question}
        Answer: 
        """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    rag_chain = (
        {"context": docsearch.as_retriever(),  "question": RunnablePassthrough()} 
        | prompt 
        | llm
        | StrOutputParser() 
    )



  
  
