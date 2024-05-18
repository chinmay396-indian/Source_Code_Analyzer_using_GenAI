import os
from git import Repo
from langchain.text_splitter import Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings


#Cloning the github repo

def repo_ingestion(repo_url):
    os.makedirs(name="repo", exist_ok=True)
    repo_path = "repo/"
    Repo.clone_from(repo_url, to_path = repo_path)

    return repo_path


#Loading repos as documents
def loading_repo_as_documents(repo_path):
    loader = GenericLoader.from_filesystem(repo_path+"/src/mlProject",
                                  glob="**\*",
                                  suffixes=[".py"],
                                  parser=LanguageParser(language=Language.PYTHON, parser_threshold=500))
    documents = loader.load()
    return documents

#creating text chunks
def chunk_documents(documents):
    doc_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON,
                                                 chunk_size = 2000,
                                                 chunk_overlap = 200)
    chunks = doc_splitter.split_documents(documents)

    return chunks

#loading Embedding Model
def load_embedding_model():
    embeddings = HuggingFaceEmbeddings()
    return embeddings


    

