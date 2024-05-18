from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Chroma


def vectorize(chunks, embeddings):

    load_dotenv()

    hf_token = os.environ.get("HF_TOKEN")
    os.environ["HF_TOKEN"] = hf_token

    '''documents = loading_repo_as_documents('/repo')
    chunks = chunk_documents(documents)
    embeddings = load_embedding_model()'''

    vector_db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="/data")
    

    print(vector_db._persist_directory)

    return vector_db





