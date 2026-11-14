"""
Streamlit web page for the basic RAG app.

Run it with:   uv run streamlit run app.py
"""

import os

import streamlit as st

from fake_docs import DOCUMENTS
from rag_core import CHAT_MODEL, build_rag_graph

st.set_page_config(page_title="Acme Cloud RAG", page_icon="📄")
st.title("Acme Cloud help assistant")
st.caption(f"Basic RAG with LangGraph · model: {CHAT_MODEL} · fake company data")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Add it to the .env file and restart the app.")
    st.stop()


def show(text: str):
    # Escape "$" so Streamlit does not read prices like $10 ... $25 as a math formula.
    st.markdown(text.replace("$", "\\$"))


@st.cache_resource
def get_app():
    # Built once: embeds the fake documents and compiles the graph.
    return build_rag_graph()


with st.sidebar:
    st.subheader("Knowledge base")
    for d in DOCUMENTS:
        with st.expander(d["title"]):
            show(d["text"])

question = st.text_input(
    "Ask a question about Acme Cloud",
    placeholder="How many days of paid leave do employees get?",
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving documents and generating the answer..."):
        result = get_app().invoke({"question": question})

    st.subheader("Answer")
    show(result["answer"])

    st.subheader("Retrieved passages")
    for doc in result["context"]:
        st.markdown(f"**{doc.metadata['title']}** (`{doc.metadata['id']}`)")
        show(doc.page_content)
