import os
from dotenv import load_dotenv, find_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch

load_dotenv(find_dotenv())

CHROMA_PATH = os.getenv("CHROMA_PATH")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

# Initialize embeddings (required by Chroma even if not used for search here)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Initialize ChromaDB
chroma_db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
    collection_name=CHROMA_COLLECTION_NAME,
)

# Get the number of records
count = chroma_db._collection.count()

print(f"Количество записей в ChromaDB (коллекция '{CHROMA_COLLECTION_NAME}'): {count}")
