import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.config import Settings
from PyPDF2 import PdfReader
import textwrap
import google.generativeai as genai

# ====== Setup API Key ======
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])  # Use secrets.toml or hardcode

# ====== ChromaDB Setup ======
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection(name="rag_pdf_app", embedding_function=embedding_function)

# ====== Helpers ======

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def chunk_text(text, chunk_size=500):
    return textwrap.wrap(text, chunk_size)

def embed_chunks(chunks):
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)

def retrieve_context(question, n_results=3):
    results = collection.query(query_texts=[question], n_results=n_results)
    return "\n".join(results["documents"][0])

def generate_answer_gemini(question, context):
    prompt = f"""You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:"""

    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text.strip()

# ====== Streamlit UI ======

st.title("📄🔍 PDF RAG App with ChromaDB + Gemini")

pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

if pdf_file:
    with st.spinner("Extracting and embedding..."):
        text = extract_text_from_pdf(pdf_file)
        chunks = chunk_text(text)
        #collection.delete()
        #all_ids = collection.get()['ids']
        #collection.delete(ids=all_ids)
        embed_chunks(chunks)
        st.success(f"Processed {len(chunks)} chunks from PDF.")

if collection.count() > 0:
    query = st.text_input("Ask a question about the PDF:")

    if query:
        with st.spinner("Thinking with Gemini..."):
            context = retrieve_context(query)
            answer = generate_answer_gemini(query, context)
            st.markdown("### 🧠 Answer")
            st.write(answer)
            with st.expander("📚 Retrieved context"):
                st.write(context)
