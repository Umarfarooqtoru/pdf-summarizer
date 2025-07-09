import os
import requests
from dotenv import load_dotenv
import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer, util

# Load .env locally
load_dotenv()

# First try Streamlit secrets (for Streamlit Cloud), fallback to .env
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

# Chunk text
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Embed chunks
def embed_chunks(chunks):
    return embedder.encode(chunks, convert_to_tensor=True)

# Semantic search
def search(query, chunks, chunk_embeddings, top_k=5):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, chunk
