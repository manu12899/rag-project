from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Step 1 - Load document
print("Loading document...")
with open("sample.txt", "r") as f:
    text = f.read()

# Step 2 - Split into chunks
print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.create_documents([text])
print(f"Created {len(chunks)} chunks")

# Step 3 - Create embeddings and FAISS index
print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
print("FAISS index created!")

# Step 4 - Connect to Groq
print("Connecting to Groq...")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Step 5 - Retrieve relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
question = "How long is the onboarding program?"
docs = retriever.invoke(question)
context = "\n".join([d.page_content for d in docs])

# Step 6 - Get answer
prompt = f"Answer this question using the context below.\n\nContext:\n{context}\n\nQuestion: {question}"
response = llm.invoke(prompt)

print(f"\n✅ RAG Pipeline Working!\n")
print(f"Question: {question}")
print(f"Answer: {response.content}")