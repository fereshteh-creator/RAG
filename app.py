import streamlit as st
from transformers import pipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="Swiss Law RAG", page_icon="⚖️")
st.title("⚖️ Swiss Law Question Answering")

@st.cache_resource
def load_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("OR_vectorstore", embeddings)
    llm = pipeline("text-generation", model="google/flan-t5-base")  # Fast + public
    return vectorstore, llm

vectorstore, llm = load_pipeline()

query = st.text_input("❓ Stelle deine juristische Frage:")

if query:
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""Beantworte die folgende Frage basierend auf dem bereitgestellten Kontext.

    Kontext:
    {context}

    Frage: {query}
    Antwort:"""

    response = llm(prompt, max_new_tokens=300, temperature=0.3)[0]['generated_text']
    st.markdown("### ✅ Antwort:")
    st.write(response)
