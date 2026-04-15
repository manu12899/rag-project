from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# Step 1 - Load document
print("Loading document...")
with open("sample.txt", "r") as f:
    text = f.read()

# Step 2 - Split into chunks
print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.create_documents([text])
texts = [c.page_content for c in chunks]
print(f"Created {len(chunks)} chunks")

# Step 3 - Build FAISS index (dense)
print("Building FAISS index...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Step 4 - Build BM25 index (sparse)
print("Building BM25 index...")
tokenized = [t.lower().split() for t in texts]
bm25 = BM25Okapi(tokenized)
print("Both indexes ready!")

# Step 5 - Hybrid search function
def hybrid_search(query, top_k=3):
    # Dense search (FAISS)
    dense_results = vectorstore.similarity_search(query, k=top_k*2)
    
    # Sparse search (BM25)
    tokens = query.lower().split()
    bm25_scores = bm25.get_scores(tokens)
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:top_k*2]
    
    # Combine using Reciprocal Rank Fusion
    scores = {}
    k = 60
    for rank, doc in enumerate(dense_results):
        key = doc.page_content[:80]
        scores[key] = scores.get(key, {"doc": doc, "score": 0})
        scores[key]["score"] += 1 / (rank + k)

    for rank, idx in enumerate(top_bm25_idx):
        key = texts[idx][:80]
        scores[key] = scores.get(key, {"doc": chunks[idx], "score": 0})
        scores[key]["score"] += 1 / (rank + k)

    # Sort by combined score
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [r["doc"] for r in ranked[:top_k]]

# Step 6 - Ask question using hybrid search
print("\n✅ Hybrid RAG Ready!\n")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

question = "How many days of annual leave do employees get?"
docs = hybrid_search(question)

print("📄 Retrieved chunks:")
print("="*50)
for i, doc in enumerate(docs):
    print(f"\nChunk {i+1}: {doc.page_content}")
print("="*50)

context = "\n".join([d.page_content for d in docs])
prompt = f"Answer this question using the context below.\n\nContext:\n{context}\n\nQuestion: {question}"
response = llm.invoke(prompt)

print(f"\nQuestion: {question}")
print(f"Answer: {response.content}")