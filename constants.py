import os
from dotenv import load_dotenv
from chromadb.config import Settings
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl='duckdb+parquet',
    persist_directory='./db',
    anonymized_telemetry=False
)

EMBEDDINGS_MODEL = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese")
