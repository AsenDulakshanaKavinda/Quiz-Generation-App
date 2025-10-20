import sys

from langchain_classic.chains.summarize import load_summarize_chain

from exception import ProjectException
from logger import logging as log

from prompts import *

from langchain_core.documents import Document

from langchain_mistralai import ChatMistralAI


from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

import os
from dotenv import load_dotenv
load_dotenv()
MISTRAL_API_KEY=os.getenv("MISTRAL_API_KEY")

from config import *

llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.0,
)

generator_chain = load_summarize_chain(llm = llm,
                                       chain_type='refine',
                                       verbose=True,
                                       question_prompt=initial_prompt,
                                       refine_prompt=refine_prompt)


def get_vectorstore():
    if os.path.exists(VECTOR_STORE_PATH):
        embeddings = MistralAIEmbeddings(
            model=EMBED_MODEL,
        )
        vs = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vs
    else:
        raise RuntimeError("Vector store not found. Run ingest_and_index first.")

def get_relavant_docs(vectorstore, query=None, k=5):
    if query is None:
        # mode 1 - get all chunks
        return vectorstore.docstore._dict.values()
    else:
        # mode 2 - similarity search
        return vectorstore.similarity_search(query, k=k)


def generate_mcqs(vectorstore, query=None):
    try:
        docs = get_relavant_docs(vectorstore, query=query)

        if not docs:
            raise ValueError("No documents found for the given query.")

        # Convert dict_values → list
        docs = list(docs)

        # Ensure docs are Document objects
        if isinstance(docs[0], str):
            docs = [Document(page_content=d) for d in docs]

        # Pass documents directly to the refine chain
        result = generator_chain.invoke({"input_documents": docs})
        return result
    except Exception as e:
        raise ProjectException(e, sys)

def generate():
    try:
        vectorstore = get_vectorstore()
        topic = input("Topic (leave empty for all data): ").strip()

        if topic:
            topic_mcqs = generate_mcqs(vectorstore, query=topic)
            print(f"\nMCQs about '{topic}':\n", topic_mcqs)
        else:
            all_mcqs = generate_mcqs(vectorstore)
            print("MCQs from full data:\n", all_mcqs)

    except ProjectException as e:
        print("Error during MCQ generation:", e)
    except Exception as e:
        print("Unexpected error:", e)


