import os
from dotenv import load_dotenv
import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# Get Groq API key securely
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

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

# Semantic search for top relevant chunks
def search(query, chunks, chunk_embeddings, top_k=5):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, chunk_embeddings, top_k=top_k)
    results = [chunks[hit['corpus_id']] for hit in hits[0]]
    return results

# Generate answer using Groq Mixtral model
def generate_groq_answer(query, context):
    prompt = f"""
You are a highly knowledgeable assistant. Use the context below to answer the user's question clearly, accurately, and in detail.

Context:
{context}

Question: {query}
Answer:
"""
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("📄 ChatPDF AI (Groq Mixtral)")

pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

if pdf_file:
    text = extract_text_from_pdf(pdf_file)
    st.success("✅ PDF text extracted.")

    chunks = chunk_text(text)
    chunk_embeddings = embed_chunks(chunks)
    st.success("✅ PDF chunks embedded.")

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
