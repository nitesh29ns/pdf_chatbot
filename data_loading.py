import os
# if not os.environ.get("NOMIC_API_KEY"):
  # os.environ["NOMIC_API_KEY"] = "nk-2qWDlRSfhpW2XxhDackPfJxBMJwVfFoxx_Hy-ae5dxE"
from langchain_nomic import NomicEmbeddings  
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from uuid import uuid4
import streamlt as st
NOMIC_API_KEY = st.secrets.NORMIC_API_KEY

if not NOMIC_API_KEY:
    raise RuntimeError("NOMIC_API_KEY is not set")

# using nomic embeddings
def get_embedding_funcation():
        embedding = NomicEmbeddings(
            model="nomic-embed-text-v1.5",)

        return embedding

# converting text into chuncks and vecterised
class vectordb:

    def __init__(self,pdf_path:str, chroma_path:str):
        try:
            self.pdf_path = pdf_path
            self.chroma_path = chroma_path
        except Exception as e:
            raise e
    
    # read the pdf
    def load_documents(self):
        try:
            document_loader = PyPDFLoader(self.pdf_path)
            return document_loader.load()
        except Exception as e:
            raise e
    
    # split data into chunks
    def split_documents(self,documents:list):
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 800,
                chunk_overlap = 80,
                length_function = len,
                is_separator_regex = False,
            )

            return text_splitter.split_documents(documents)
        
        except Exception as e:
            raise e

    # add data to chroma DB
    def add_to_chroma(self,chunks:list):
        try:
            db = Chroma(
                persist_directory = self.chroma_path,
                embedding_function = get_embedding_funcation() 
            )

            if len(chunks):
                print(f"➕ Adding new documents: {len(chunks)}")
                chunks_ids = [str(uuid4()) for _ in range(len(chunks))]
                db.add_documents(chunks, ids=chunks_ids)
                return "vector db is created."
            else:
                print("No new documents added.")
            
        except Exception as e:
            raise e
        
        
    def upload_to_vectordb(self):
        try:
            documents = self.load_documents()
            chunks = self.split_documents(documents)
            self.add_to_chroma(chunks)
            return "db created."
        except Exception as e:
            raise e
        

"""
dd = vectordb(pdf_path="D:\ml\PDF_RAG_llm\pdf_chatbot\data\monopoly.pdf",chroma_path="./test1")

d = dd.upload_to_vectordb()

print(d)
"""
