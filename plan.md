## **Project Goal**

Develop a production-ready, serverless RAG system that provides high-precision citations for a 50-year General Conference corpus while maintaining a **$0/month infrastructure cost.**

---

## **1. Technical Architecture & Stack**

- **Orchestrator:** `LlamaIndex` using a `RecursiveRetriever` for hierarchical mapping.
- **Vector Database:** `Pinecone (Serverless)` on **GCP** (storing 1536-dim Child vectors).
- **Context Storage:** `SimpleDocumentStore` exported as a **JSON file** (storing 1536-token Parent text for free).
- **AI Models:** * `text-embedding-3-small` (OpenAI) for cost-efficient vectorization.
  - `Cohere Rerank v3` for precision filtering.
  - `gpt-4o-mini` for final answer synthesis.
- **Frontend & Hosting:** `Streamlit` + `Streamlit Community Cloud`.

---

## **2. Phase 1: The Ingestion Pipeline (**`ingest.py`**)**

This script is run locally once to prep your data for the cloud.

1. **Data Preprocessing:** Load the 50-year CSV and standardize the "Speaker," "Year," and "Title" metadata.
2. **Hierarchical Chunking:**
  - **Parents:** ~1536 tokens. These capture the full story or doctrinal argument.
  - **Children:** ~256 tokens with a 50-token overlap. These are small "needles" for the search engine to find.
3. **Mapping & Storage:**
  - Link every Child vector to its Parent ID.
  - **Upload:** Push Child vectors to Pinecone.
  - **Export:** Save Parent text as a JSON file in the `storage/` directory (committed to GitHub).

---

## **3. Phase 2: Retrieval Intelligence (**`app.py`**)**

This logic executes in real-time when a user asks a question.

- **Recursive Retrieval:** The system queries Pinecone for the top **Child** matches but immediately "swaps" them for the **1536-token Parent** text from your JSON vault.
- **Reranking Layer:** Pull the top 20 candidate chunks and use **Cohere Rerank** to select the top 3-5 most relevant segments.
- **System Prompting:** Instruct the LLM to act as a research assistant, using *only* the provided text and citing the Speaker and Year for every claim.

---

## **4. Phase 3: UI, Security, & Deployment**

- **Chat Interface:** Use `st.chat_message` for a modern UI. Include an "expandable" section under each response to show the source material.
- **Gatekeeping:** Implement a simple password entry in the sidebar to protect your API keys from public bot traffic.
- **Deployment:** * Push code and the `parent_store.json` to GitHub.
  - Deploy via Streamlit Community Cloud.
  - Add API keys (OpenAI, Pinecone, Cohere) to the **Streamlit Secrets** dashboard.

---

## **5. Phase 4: Validation (Portfolio Differentiator)**

- **RAGAS Evaluation:** Create 10 "Golden Questions" and run an automated audit to score **Faithfulness** (no hallucinations) and **Answer Relevance**.
- **Documentation:** Write a README that highlights the "Parent-Child" logic and the cost-conscious decision to use JSON over a managed SQL database.

