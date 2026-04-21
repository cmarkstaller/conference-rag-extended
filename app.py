"""
General Conference RAG — Streamlit query app (Pinecone + Cohere rerank + OpenAI).

Where things run / are hosted
------------------------------
**On your laptop (or Streamlit Community Cloud when you deploy):**
- This Streamlit app (Python process). It orchestrates retrieval and answer generation.

**In the cloud (managed services you call over HTTPS):**
- **Pinecone**: stores *child chunk vectors* only. Each hit includes metadata (parent_id,
  title, speaker, year, url, raw child `text`). No LLM runs here — it is vector search.
- **OpenAI**: (1) *embeddings* for the user question — same model as ingest, typically
  `text-embedding-3-small`. (2) *chat completion* for the final answer, e.g. `gpt-4o-mini`.
- **Cohere**: *reranking* only — re-orders candidate *parent* passages for the LLM.
  Your API key is used from `.env` locally or Streamlit Secrets in production.

**In your repo / disk (not “hosted” as a server):**
- `storage/parent_store.json`: *parent chunk text* keyed by `parent_id`. The app loads
  this file into memory (cached) to swap child hits → full parent context (small-to-big).

**What Streamlit Cloud hosts (optional later):**
- Only the Streamlit app process. It still talks to Pinecone, OpenAI, and Cohere over the
  internet. You add the same API keys in Streamlit’s Secrets UI — they are not embedded
  in the GitHub repo.

Environment variables
---------------------
Required:
  OPENAI_API_KEY
  PINECONE_API_KEY
  COHERE_API_KEY

Optional (defaults shown):
  PINECONE_INDEX=conference-rag
  PINECONE_NAMESPACE=talks-v1
  PARENT_STORE_PATH=storage/parent_store.json
  EMBEDDING_MODEL=text-embedding-3-small
  CHAT_MODEL=gpt-4o-mini
  COHERE_RERANK_MODEL=rerank-english-v3.0
  PINECONE_TOP_K=20
  RERANK_TOP_N=8
  LLM_CONTEXT_CHUNKS=5
  APP_PASSWORD=   # if set, user must enter this in the sidebar before chatting

Run locally:
  streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv
from components.chat_ui import render_chat_history, render_sources_expander
from services.rag_pipeline import RagService, load_parent_store
from utils.config import load_app_config


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="General Conference RAG", page_icon="assets/talkfinderlogo.png", layout="wide")
    st.markdown(
        "<h3 style='text-align:center; font-weight:500; margin-bottom: 0.25rem;'>Search General Conference for your question</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#6b7280; font-size:0.95rem; margin-top:0; margin-bottom:1rem;'>searching talks from 1971 onward</p>",
        unsafe_allow_html=True,
    )

    config = load_app_config()
    if not config.openai_key or not config.pinecone_key or not config.cohere_key:
        st.error("Missing one or more API keys. Set OPENAI_API_KEY, PINECONE_API_KEY, and COHERE_API_KEY in `.env` or Streamlit Secrets.")
        st.stop()

    with st.sidebar:
        mode = st.radio("Mode", options=["Chat answer", "Talk finder"], index=0)
        st.subheader("Settings")
        st.text_input("Pinecone index", value=config.index_name, disabled=True)
        st.text_input("Namespace", value=config.namespace, disabled=True)
        st.text_input("Parent store", value=config.parent_path, disabled=True)
        st.number_input("Pinecone top_k", value=config.pinecone_top_k, disabled=True, help="Set PINECONE_TOP_K in env to change")
        st.number_input("Rerank pool size", value=config.rerank_top_n, disabled=True, help="Set RERANK_TOP_N in env to change")
        st.number_input("Chunks to LLM", value=config.llm_chunks, disabled=True, help="Set LLM_CONTEXT_CHUNKS in env to change")

        unlocked = True
        if config.app_password:
            entered = st.text_input("App password", type="password")
            unlocked = entered == config.app_password
            if not unlocked:
                st.warning("Enter the correct password to use the chat.")

    if config.app_password and not unlocked:
        st.stop()

    parent_store = load_parent_store(config.parent_path)
    if not parent_store:
        st.warning(
            f"No parent store loaded from `{config.parent_path}`. Answers will fall back to child snippets only "
            "unless you generate `parent_store.json` with `ingest.py`."
        )

    rag_service = RagService(config)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if mode == "Chat answer":
        render_chat_history(st.session_state.messages)

        if prompt := st.chat_input("ask anything..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving and generating…"):
                    reply, sources, warning_message = rag_service.run_query(prompt, parent_store)
                    if warning_message:
                        st.warning(warning_message)

                    st.markdown(reply)
                    if sources:
                        render_sources_expander(sources, title="Sources used")

            st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})

    else:
        st.markdown(
            "<p style='text-align:center;'>Find the most relevant talks to your question</p>",
            unsafe_allow_html=True,
        )

        if "talk_search_visible_count" not in st.session_state:
            st.session_state.talk_search_visible_count = 15
        if "talk_search_results" not in st.session_state:
            st.session_state.talk_search_results = []
        if "talk_search_query" not in st.session_state:
            st.session_state.talk_search_query = ""

        if query := st.chat_input("Search talks (e.g., faith in difficult times)"):
            with st.spinner("Searching vector database..."):
                st.session_state.talk_search_results = rag_service.search_talks(query.strip(), top_k=200)
                st.session_state.talk_search_visible_count = 15
                st.session_state.talk_search_query = query.strip()

        results = st.session_state.talk_search_results
        if results:
            st.caption(
                f"Showing {min(st.session_state.talk_search_visible_count, len(results))} of {len(results)} talks "
                f"for: {st.session_state.talk_search_query}"
            )
            for i, talk in enumerate(results[: st.session_state.talk_search_visible_count], start=1):
                speaker = talk.get("speaker", "")
                title = talk.get("title", "")
                year = talk.get("year", "")
                season = talk.get("season", "")
                url = talk.get("url", "")
                score = talk.get("score", 0.0)
                st.markdown(
                    f"**{i}. {speaker} — {title}**  \n"
                    f"{year} {season}  \n"
                    f"Relevance score: `{score:.4f}`  \n"
                    f"[{url}]({url})"
                )

            if st.session_state.talk_search_visible_count < len(results):
                if st.button("Load 15 more"):
                    st.session_state.talk_search_visible_count += 15
                    st.rerun()
        elif st.session_state.talk_search_query:
            st.info("No talks found for that query.")


if __name__ == "__main__":
    main()
