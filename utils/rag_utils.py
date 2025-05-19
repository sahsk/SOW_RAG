from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

def chunk_texts(texts, chunk_size=1000, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs = []
    for t in texts:
        docs.extend(splitter.create_documents([t]))
    return docs

def build_vector_db(docs, api_key):
    if not api_key:
        raise ValueError("OpenAI API key is missing! Pass the API key directly.")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.from_documents(docs, embeddings)
    return db

def retrieve_top_k(db, query, k=5):
    retriever = db.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return [d.page_content for d in docs]
