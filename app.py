import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Load models
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    return embedder, qa_pipeline

embedder, qa_pipeline = load_models()

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
def search(query, chunks, chunk_embeddings, top_k=3):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, chunk_embeddings, top_k=top_k)
    results = [chunks[hit['corpus_id']] for hit in hits[0]]
    return results

# Answer generation
def generate_answer(query, context):
    result = qa_pipeline(question=query, context=context)
    return result['answer']

# Streamlit UI
st.title("📄 Chat with Your PDF (Hugging Face)")

pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

if pdf_file:
    text = extract_text_from_pdf(pdf_file)
    st.success("✅ PDF text extracted.")

    chunks = chunk_text(text)
    chunk_embeddings = embed_chunks(chunks)
    st.success("✅ PDF chunks embedded.")

    query = st.text_input("Ask something about your PDF:")
    if query:
        top_chunks = search(query, chunks, chunk_embeddings, top_k=3)
        context = " ".join(top_chunks)
        answer = generate_answer(query, context)

        st.write("### 🔎 Top Relevant Chunks:")
        for c in top_chunks:
            st.write(f"- {c[:300]}...")  # Show snippet

        st.write("### 📝 Answer:")
        st.write(answer)
