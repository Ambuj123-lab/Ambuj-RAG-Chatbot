# 🦁 Ambuj's Secure RAG AI Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ambuj-rag-chatbot.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/🦜🔗-LangChain-green)](https://python.langchain.com/)
[![Llama 3.3](https://img.shields.io/badge/🤖-Llama_3.3-purple)](https://ai.meta.com/llama/)
[![Security](https://img.shields.io/badge/🛡️-Enterprise_Security-red)](https://microsoft.github.io/presidio/)
[![Redis](https://img.shields.io/badge/Redis-Analytics-red)](https://upstash.com/)

> **"Not just a Chatbot, but a Secure, Self-Healing Enterprise AI Solution."**

## 🚀 Live Demo
**[Click Here to Interact with the Bot](https://ambuj-rag-chatbot.streamlit.app/)**  
*(Login Required: Ask Admin for Access Key)*

---

## 🌟 Why This Project Stands Out? (The "X-Factor")

Most developers build simple RAG bots. This project demonstrates **Enterprise-Grade Engineering** capabilities:

### 1. 🛡️ PII Masking (The Security Pro)
> *Privacy First Architecture*
Instead of blindly sending user data to the LLM, this system implements a **Privacy Layer** using **Microsoft Presidio & SpaCy**. It detects and masks sensitive info (Names, Phones, Emails) *before* it leaves the secure environment.
*   **Why it matters:** Shows deep understanding of **GDPR/Data Privacy** and Enterprise Security requirements.

### 2. 🔧 Cross-Platform Engineering (The DevOps Mindset)
> *Write on Windows, Deploy on Linux*
A common failure point in RAG apps is the "Path Mismatch" (Windows `\` vs Linux `/`) which crashes databases on the cloud. This system uses **OS-Agnostic Path Handling** (`os.path.join`) and robust error handling to ensure seamless deployment across environments.
*   **Why it matters:** Demonstrates **DevOps** awareness and robust coding practices.

### 3. 🏗️ Self-Healing RAG (The Reliable Architect)
> *Zero-Downtime Knowledge Base*
Databases on free cloud tiers often get corrupted or deleted on reboot. This system features an **Auto-Recovery Mechanism**:
*   If the Vector DB is missing or corrupt, the system detects it.
*   It automatically **rebuilds the index** from the persistent `data/` source in seconds.
*   **Why it matters:** Proves ability to design **Fault-Tolerant Systems**.

---

## ⚡ Key Features

*   **🧠 Hybrid Knowledge Base:**
    *   **Permanent Memory:** Pre-loaded with "Ambuj's Resume" & "Consumer Protection Act".
    *   **Dynamic Learning:** Users can upload their own PDFs (e.g., Job Descriptions) to "Challenge" the bot.
*   **⚡ Incremental Indexing:** Smart logic that adds *only* new files to the database in **<2 seconds**, avoiding slow full rebuilds.
*   **📊 Real-Time Analytics:** Integrated **Redis (Upstash)** to track unique visitor counts securely.
*   **💬 Multi-Lingual Support:** Automatically detects and responds in **Hindi (Devanagari)**, **Hinglish**, or **English** based on user input.
*   **🚫 Hallucination Control:** Strict system prompts ensure the bot says "I don't know" instead of making up facts, ensuring **Information Integrity**.
*   **🎨 Premium UI:** Glassmorphism design, animated gradients, and a custom login portal.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **LLM** | **Llama 3.3 (70B)** | via OpenRouter API for high-quality reasoning |
| **Orchestration** | **LangChain** | For RAG pipeline and prompt management |
| **Vector DB** | **ChromaDB** | For storing and retrieving document embeddings |
| **Embeddings** | **HuggingFace** | `all-MiniLM-L6-v2` for efficient semantic search |
| **Security** | **Presidio + SpaCy** | For PII detection and anonymization |
| **Frontend** | **Streamlit** | With custom CSS for premium aesthetics |
| **Database** | **Upstash Redis** | For visitor analytics and session management |

---

## 📂 Project Structure

```bash
My_RAG_Bot/
├── 📄 app.py                # Main Application Logic (The Brain)
├── 📄 requirements.txt      # Dependencies (Locked versions for stability)
├── 📄 .env                  # Secrets (API Keys - Not pushed to GitHub)
├── 📂 data/                 # Permanent Knowledge Source (PDFs)
│   ├── ambuj_resume.pdf
│   └── consumer_act.pdf
└── 📂 chroma_db/            # Vector Database (Auto-generated on Cloud)
```

## 👨‍💻 Developer

**Ambuj Kumar Tripathi**  
*GenAI Engineer | Solution Architect | Prompt Specialist*

---
*Built with ❤️ and ☕ using Python.*
