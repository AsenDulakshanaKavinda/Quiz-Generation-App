# Script to load docs → split → embed → store
import sys
from exception import ProjectException
from logger import logging as log

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

import os
from dotenv import load_dotenv
load_dotenv()
MISTRAL_API_KEY=os.getenv("MISTRAL_API_KEY")

from config import *

def data_ingest_and_index(pdf_paths: list, persist: bool = True):
    # load all the docs
    all_docs = []
    for path in pdf_paths:
        if path.lower().endswith('.pdf'):
            loader = PyPDFLoader(path)
            docs = loader.load()
        else:
            loader = TextLoader(path, encoding='utf-8')
            docs = loader.load()
        log.info(f"Data loading from {path} complited...")
    all_docs.extend(docs)
    log.info("Data loading complited...")

    spliter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators = SEPARATORS
    )

    chunks = []
    for d in all_docs:
        for i, c in enumerate(spliter.split_documents([d])):
            c.metadata["source_doc"] = os.path.basename(d.metadata.get("source", path))
            c.metadata["chunk_id"] = f"{c.metadata['source_doc']}_chunk{i}"
            chunks.append(c)
    log.info("Data chunking complited...")

    embeddings = MistralAIEmbeddings(
            model=EMBED_MODEL,
        )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    log.info("Data embedding complited...")

    if persist:
        vectorstore.save_local(VECTOR_STORE_PATH)
        log.info(f"Vector data storing in {VECTOR_STORE_PATH} complited...")
    return vectorstore

if __name__ == "__main__":
    pdf_paths = ['../data/test_docs/test01.pdf', 
                 '../data/test_docs/test02.pdf', 
                 '../data/test_docs/test03.pdf']
    data_ingest_and_index(pdf_paths)










