import streamlit as st
import fitz
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Functions
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text()
    return text

@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return summarizer, embedder

def summarize_text(text, summarizer, max_chunk=1024):
    summaries = []
    for i in range(0, len(text), max_chunk):
        chunk = text[i:i+max_chunk]
        summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return summaries

def search_query(query, summaries, embedder, top_k=3):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    summary_embeddings = embedder.encode(summaries, convert_to_tensor=True)
    
    hits = util.semantic_search(query_embedding, summary_embeddings, top_k=top_k)
    results = []
    for hit in hits[0]:
        results.append(summaries[hit['corpus_id']])
    return results

# Streamlit UI
st.title("📄 PDF Search Summarizer")

pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

if pdf_file:
    summarizer, embedder = load_models()
    text = extract_text_from_pdf(pdf_file)
    st.success("✅ PDF text extracted.")

    summaries = summarize_text(text, summarizer)
    st.success("✅ Text summarized.")

    query = st.text_input("Enter your search query:")
    if query:
        results = search_query(query, summaries, embedder)
        st.write("### 🔎 Search Results:")
        for r in results:
            st.write(f"- {r}")
