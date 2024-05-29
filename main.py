# main.py
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFaceHub
import pinecone
from dotenv import load_dotenv
import os

class ChatBot:
    def __init__(self):
        load_dotenv()
        self.loader = TextLoader('./horoscope.txt')
        documents = self.loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        self.docs = text_splitter.split_documents(documents)

        self.embeddings = HuggingFaceEmbeddings()

        pinecone.init(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment='gcp-starter'
        )

        self.index_name = "langchain-demo"

        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(name=self.index_name, metric="cosine", dimension=768)
            self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, index_name=self.index_name)
        else:
            self.docsearch = Pinecone.from_existing_index(self.index_name, self.embeddings)

        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50},
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

        from langchain import PromptTemplate

        self.template = """
        You are a seer. These Human will ask you questions about their life. Use the following piece of context to answer the question. 
        If you don't know the answer, just say you don't know. 
        You answer with a short and concise answer, no longer than 2 sentences.

        Context: {context}
        Question: {question}
        Answer: 
        """

        self.prompt = PromptTemplate(template=self.template, input_variables=["context", "question"])

        from langchain.schema.runnable import RunnablePassthrough
        from langchain.schema.output_parser import StrOutputParser

        self.rag_chain = (
            {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def get_response(self, question):
        response_dict = self.rag_chain.invoke({"question": question})
        return response_dict['Answer']  # Assuming the response is returned with the key 'Answer'

