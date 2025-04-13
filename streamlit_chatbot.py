import streamlit as st
import os
from scripts.data_loader import load_and_prepare_data
from scripts.embedder import generate_embeddings, create_faiss_index
from scripts.retriever import query_lessons

st.set_page_config(page_title="Lessons Learnt Chatbot", layout="wide")
st.title("📘 Lessons Learnt Chatbot")
st.markdown("Ask a question to explore past project experiences and insights.")

uploaded_file = st.file_uploader("Upload Lessons Excel File", type=["xlsx"])

@st.cache_data(show_spinner=False)
def setup_pipeline(filepath):
    texts = load_and_prepare_data(filepath)
    embeddings, model = generate_embeddings(texts)
    index, id_to_text = create_faiss_index(embeddings, texts)
    return model, index, id_to_text

if uploaded_file:
    file_path = f"data/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    model, index, id_to_text = setup_pipeline(file_path)

    query = st.text_input("Your question:", placeholder="e.g. What caused cable sizing issues?")

    if query:
        with st.spinner("Searching lessons..."):
            results = query_lessons(query, model, index, id_to_text)
        for i, res in enumerate(results, 1):
            st.markdown(f"### Result {i}")
            st.code(res)
else:
    st.info("Please upload an Excel file to start.")
