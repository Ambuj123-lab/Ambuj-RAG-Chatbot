import streamlit as st
import os
import re
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyMuPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")


def detect_language(text):
    """Detect if input is Hindi or English."""
    hindi_chars = re.findall(r'[\u0900-\u097F]', text)
    if len(hindi_chars) > 2:
        return "hindi"
    return "english"


st.set_page_config(page_title="Ambuj's RAG Bot", layout="wide")
st.title("🤖 Ambuj's AI Assistant")
st.markdown(
    "Chat with your documents in Hindi or English!"
)

# Sidebar for document processing
with st.sidebar:
    st.header("📂 Document Manager")
    if st.button("🚀 Process Documents"):
        with st.spinner("Processing PDFs..."):
            try:
                if not os.path.exists("data"):
                    os.makedirs("data")
                    st.error(
                        "Created 'data' folder. Please add PDFs."
                    )
                    st.stop()

                loader = DirectoryLoader(
                    "./data", glob="*.pdf", loader_cls=PyMuPDFLoader
                )

                try:
                    documents = loader.load()
                except Exception as e:
                    st.error(f"Error loading PDFs: {e}")
                    st.stop()

                if not documents:
                    st.error("No PDFs found in 'data' folder!")
                    st.stop()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                chunks = text_splitter.split_documents(documents)

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                if os.path.exists("./chroma_db"):
                    shutil.rmtree("./chroma_db")

                vector_db = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="./chroma_db"
                )
                st.success(
                    f"✅ Indexed {len(chunks)} chunks successfully!"
                )

            except Exception as e:
                st.error(f"Error: {e}")

# Chat interface
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_input := st.chat_input("Ask a question..."):
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vector_db = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings
            )
            retriever = vector_db.as_retriever(
                search_kwargs={"k": 3}
            )
            relevant_docs = retriever.invoke(user_input)

            context_text = "\n\n".join([
                f"Source: {doc.metadata.get('source', 'Unknown')}\n"
                f"Content: {doc.page_content}"
                for doc in relevant_docs
            ])

            if not context_text:
                fallback_msg = (
                    "No relevant documents found. "
                    "Please process documents first."
                )
                st.markdown(fallback_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": fallback_msg
                })
                st.stop()

            lang = detect_language(user_input)

            if lang == "hindi":
                system_prompt = (
                    "Aap ek helpful AI assistant ho. "
                    "Jawab sirf context se dena. "
                    "Hindi ya Hinglish mein jawab dena.\n\n"
                    "Context:\n{context}\n\n"
                    "Question:\n{question}"
                )
            else:
                system_prompt = (
                    "You are a helpful AI assistant. "
                    "Answer ONLY based on context. "
                    "Respond in English.\n\n"
                    "Context:\n{context}\n\n"
                    "Question:\n{question}"
                )

            prompt_template = ChatPromptTemplate.from_template(
                system_prompt
            )
            llm = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=API_KEY,
                model="meta-llama/llama-3-8b-instruct:free",
                temperature=0.3,
                streaming=True
            )

            parser = StrOutputParser()
            chain = prompt_template | llm | parser

            response = st.write_stream(
                chain.stream({
                    "context": context_text,
                    "question": user_input
                })
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

            st.markdown("---")
            with st.expander("📄 Source Documents"):
                for i, doc in enumerate(relevant_docs):
                    source = doc.metadata.get('source', 'Unknown')
                    st.write(f"**Source {i+1}:** {source}")
                    st.text(doc.page_content[:300] + "...")

        except Exception as e:
            st.error(f"Error: {e}")
