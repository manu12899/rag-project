from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os

load_dotenv()

app = FastAPI()

print("Setting up indexes...")
with open("sample.txt", "r") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.create_documents([text])
texts = [c.page_content for c in chunks]

embeddings = FakeEmbeddings(size=384)
vectorstore = FAISS.from_documents(chunks, embeddings)

tokenized = [t.lower().split() for t in texts]
bm25 = BM25Okapi(tokenized)
print("Setup complete!")

def get_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    return ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile", temperature=0)

def hybrid_search(query, top_k=3):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k * 2})
    dense_docs = retriever.invoke(query)
    tokens = query.lower().split()
    bm25_scores = bm25.get_scores(tokens)
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:top_k * 2]
    scores = {}
    k = 60
    for rank, doc in enumerate(dense_docs):
        key = doc.page_content[:80]
        scores[key] = scores.get(key, {"doc": doc, "score": 0})
        scores[key]["score"] += 1 / (rank + k)
    for rank, idx in enumerate(top_bm25_idx):
        key = texts[idx][:80]
        scores[key] = scores.get(key, {"doc": chunks[idx], "score": 0})
        scores[key]["score"] += 1 / (rank + k)
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [r["doc"] for r in ranked[:top_k]]

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "RAG System is running!"}

@app.post("/ask")
def ask(request: QueryRequest):
    llm = get_llm()
    query = request.question
    docs = hybrid_search(query)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"Answer this question using the context below.\n\nContext:\n{context}\n\nQuestion: {query}"
    response = llm.invoke(prompt)
    return {
        "question": query,
        "answer": response.content
    }