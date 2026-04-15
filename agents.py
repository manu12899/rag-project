from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# ── Setup ──────────────────────────────────────────────
print("Setting up indexes...")
with open("sample.txt", "r") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.create_documents([text])
texts = [c.page_content for c in chunks]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

tokenized = [t.lower().split() for t in texts]
bm25 = BM25Okapi(tokenized)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
print("Setup complete!")

# ── Hybrid Search ──────────────────────────────────────
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

# ── Agent 1: Retrieval Agent ───────────────────────────
def retrieval_agent(query):
    print("\n🔍 Retrieval Agent searching...")
    docs = hybrid_search(query)
    context = "\n".join([d.page_content for d in docs])
    return context

# ── Agent 2: Summarization Agent ──────────────────────
def summarization_agent(context, query):
    print("📝 Summarization Agent summarizing...")
    prompt = f"""You are a summarization expert.
Summarize the key facts from this context that answer the query.
Keep it concise and clear.

Context: {context}
Query: {query}

Summary:"""
    response = llm.invoke(prompt)
    return response.content

# ── Agent 3: Fact-Check Agent ─────────────────────────
def factcheck_agent(answer, query):
    print("✅ Fact-Check Agent verifying...")
    docs = hybrid_search(query)
    evidence = "\n".join([d.page_content for d in docs])
    prompt = f"""You are a fact-checking expert.
Check if this answer is supported by the evidence.
Reply with SUPPORTED or UNSUPPORTED and a short reason.

Answer: {answer}
Evidence: {evidence}

Verdict:"""
    response = llm.invoke(prompt)
    return response.content

# ── Orchestrator Agent ────────────────────────────────
def orchestrator(query):
    print(f"\n🤖 Orchestrator processing: '{query}'")
    print("="*50)
    context = retrieval_agent(query)
    summary = summarization_agent(context, query)
    verdict = factcheck_agent(summary, query)
    print("\n" + "="*50)
    print(f"📌 Question   : {query}")
    print(f"📄 Answer     : {summary}")
    print(f"🔎 Fact-Check : {verdict}")
    print("="*50)

# ── Run it! ───────────────────────────────────────────
orchestrator("How many days of annual leave do employees get?")
orchestrator("What are the office hours?")
orchestrator("How much expenses need manager approval?")