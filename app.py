import os
import requests
from dotenv import load_dotenv
import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer, util

# Load environment variables
load_dotenv()

# Get Groq API key securely
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Please set it in Streamlit Secrets or .env file.")
    st.stop()

# Load embedder model
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()

# Extract PDF text
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text()
    return text

# Chunk text into manageable pieces
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Embed chunks
def embed_chunks(chunks):
    return embedder.encode(chunks, convert_to_tensor=True)

# Semantic search
def search(query, chunks, chunk_embeddings, top_k=5):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, chunk_embeddings, top_k=top_k)
    results = [chunks[hit['corpus_id']] for hit in hits[0]]
    return results

# Generate answer using Groq LLaMA 3 via direct API call
def generate_groq_answer(query, context):
    context = context[:3000]  # Limit context
    prompt = f"""
You are a highly knowledgeable assistant. Use the context below to answer the user's question clearly, accurately, and in detail.

Context:
{context}

Question: {query}
Answer:
"""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    payload = {
        "model": "llama3-70b-8192",  # Updated to LLaMA 3-70B model
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        st.error(f"❌ Groq API error: {response.status_code} - {response.text}")
        return "Error retrieving answer."

    data = response.json()
    return data['choices'][0]['message']['content']

# Streamlit UI
st.title("📄 ChatPDF AI (Groq LLaMA 3-70B)")

pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

if pdf_file:
    text = extract_text_from_pdf(pdf_file)
    st.success("✅ PDF text extracted.")

    chunks = chunk_text(text)
    chunk_embeddings = embed_chunks(chunks)
    st.success("✅ PDF chunks embedded and ready.")

    query = st.text_input("Ask something about your PDF:")
    if query:
        top_chunks = search(query, chunks, chunk_embeddings, top_k=5)
        context = " ".join(top_chunks)
        answer = generate_groq_answer(query, context)

        st.write("### 🔎 Top Relevant Chunks:")
        for c in top_chunks:
            st.write(f"- {c[:300]}...")  # Show snippet

        st.write("### 📝 Answer:")
        st.write(answer)
