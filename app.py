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

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cohere
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone


def _secret(key: str, default: str | None = None) -> str | None:
    """Read from environment first, then Streamlit Cloud secrets."""
    v = os.getenv(key)
    if v:
        return v
    try:
        return str(st.secrets[key])
    except Exception:
        return default


@st.cache_resource
def load_parent_store(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def embed_question(openai_client: OpenAI, model: str, text: str) -> list[float]:
    r = openai_client.embeddings.create(model=model, input=[text])
    return r.data[0].embedding


def pinecone_child_search(
    index: Any,
    namespace: str,
    vector: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    """Return Pinecone matches: each has id, score, metadata dict."""
    res = index.query(
        vector=vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )
    matches = getattr(res, "matches", None) or res.get("matches", [])  # type: ignore[union-attr]
    out: list[dict[str, Any]] = []
    for m in matches:
        meta = getattr(m, "metadata", None) or m.get("metadata") or {}
        mid = getattr(m, "id", None) or m.get("id")
        score = getattr(m, "score", None) if getattr(m, "score", None) is not None else m.get("score")
        out.append({"id": mid, "score": score, "metadata": dict(meta) if meta else {}})
    return out


def ordered_unique_parent_ids(matches: list[dict[str, Any]], max_parents: int) -> list[str]:
    """Preserve best-child-hit order while de-duplicating parent_id."""
    seen: set[str] = set()
    ordered: list[str] = []
    for m in matches:
        pid = (m.get("metadata") or {}).get("parent_id")
        if not pid or not isinstance(pid, str):
            continue
        if pid in seen:
            continue
        seen.add(pid)
        ordered.append(pid)
        if len(ordered) >= max_parents:
            break
    return ordered


def build_parent_documents(
    parent_store: dict[str, Any],
    parent_ids: list[str],
    matches: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Build parallel lists: document strings for Cohere, and meta for UI / citations.

    Uses parent_store[parent_id]['text'] when present; otherwise falls back to child text
    from the best Pinecone hit for that parent.
    """
    # Map parent_id -> best match metadata (first occurrence in sorted matches)
    best_child_meta: dict[str, dict[str, Any]] = {}
    for m in matches:
        meta = m.get("metadata") or {}
        pid = meta.get("parent_id")
        if pid and pid not in best_child_meta:
            best_child_meta[str(pid)] = meta

    docs: list[str] = []
    metas: list[dict[str, Any]] = []
    for pid in parent_ids:
        rec = parent_store.get(pid)
        if rec and rec.get("text"):
            body = str(rec["text"])
            docs.append(body)
            metas.append(
                {
                    "parent_id": pid,
                    "title": rec.get("title", ""),
                    "speaker": rec.get("speaker", ""),
                    "year": rec.get("year", ""),
                    "season": rec.get("season", ""),
                    "url": rec.get("url", ""),
                }
            )
            continue
        meta = best_child_meta.get(pid, {})
        title = meta.get("title", "")
        speaker = meta.get("speaker", "")
        year = meta.get("year", "")
        season = meta.get("season", "")
        snippet = meta.get("text", "")
        fallback = f"[Speaker: {speaker}, Year: {year}, Season: {season}, Title: {title}]\n{snippet}"
        docs.append(fallback)
        metas.append(
            {
                "parent_id": pid,
                "title": title,
                "speaker": speaker,
                "year": year,
                "season": season,
                "url": meta.get("url", ""),
                "note": "parent not in local store; using child snippet from Pinecone metadata",
            }
        )
    return docs, metas


def cohere_rerank(
    cohere_client: Any,
    model: str,
    query: str,
    documents: list[str],
    top_n: int,
) -> list[tuple[int, float]]:
    """Returns list of (original_index, relevance_score) sorted best-first."""
    if not documents:
        return []
    resp = cohere_client.rerank(model=model, query=query, documents=documents, top_n=min(top_n, len(documents)))
    results = getattr(resp, "results", None) or []
    out: list[tuple[int, float]] = []
    for r in results:
        idx = int(getattr(r, "index", 0))
        score = float(getattr(r, "relevance_score", 0.0))
        out.append((idx, score))
    return out


def answer_question(
    openai_client: OpenAI,
    chat_model: str,
    user_question: str,
    contexts: list[str],
    sources: list[dict[str, Any]],
) -> str:
    joined = []
    for i, (ctx, src) in enumerate(zip(contexts, sources), start=1):
        cite = f"{src.get('speaker','')} — {src.get('title','')} ({src.get('year','')}, {src.get('season','')})"
        joined.append(f"### Source {i}\n{cite}\nURL: {src.get('url','')}\n\n{ctx}")
    system = (
        "You are a careful research assistant for General Conference talks. "
        "Answer ONLY using the provided sources. If the sources do not contain enough "
        "information, say you are not sure. For factual claims, cite the speaker and year "
        "in plain text (no fake quotes). Do not invent talks or quotes."
    )
    user = f"Question:\n{user_question}\n\nSources:\n" + "\n\n".join(joined)
    comp = openai_client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return (comp.choices[0].message.content or "").strip()


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="General Conference RAG", page_icon="assets/talkfinderlogo.png", layout="wide")
    st.title("General Conference RAG")
    st.caption("Pinecone (child vectors) → parent text (JSON) → Cohere rerank → OpenAI answer")

    with st.expander("Where is what hosted?", expanded=False):
        st.markdown(
            """
- **This app**: runs on your machine with `streamlit run`, or on **Streamlit Community Cloud** if you deploy the repo.
- **Pinecone**: hosted vector index (child chunks). Queries happen over the internet.
- **OpenAI**: embeddings + chat over HTTPS.
- **Cohere**: rerank API over HTTPS.
- **Parent text**: loaded from **`storage/parent_store.json`** on the machine running Streamlit (ship this file with the app / repo, or mount it in cloud).
            """
        )

    openai_key = _secret("OPENAI_API_KEY")
    pinecone_key = _secret("PINECONE_API_KEY")
    cohere_key = _secret("COHERE_API_KEY")
    if not openai_key or not pinecone_key or not cohere_key:
        st.error("Missing one or more API keys. Set OPENAI_API_KEY, PINECONE_API_KEY, and COHERE_API_KEY in `.env` or Streamlit Secrets.")
        st.stop()

    index_name = _secret("PINECONE_INDEX", "conference-rag") or "conference-rag"
    namespace = _secret("PINECONE_NAMESPACE", "talks-v1") or "talks-v1"
    parent_path = _secret("PARENT_STORE_PATH", "storage/parent_store.json") or "storage/parent_store.json"
    embed_model = _secret("EMBEDDING_MODEL", "text-embedding-3-small") or "text-embedding-3-small"
    chat_model = _secret("CHAT_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
    rerank_model = _secret("COHERE_RERANK_MODEL", "rerank-english-v3.0") or "rerank-english-v3.0"

    pinecone_top_k = int(_secret("PINECONE_TOP_K", "20") or "20")
    rerank_top_n = int(_secret("RERANK_TOP_N", "8") or "8")
    llm_chunks = int(_secret("LLM_CONTEXT_CHUNKS", "5") or "5")
    max_parents_for_rerank = int(_secret("MAX_PARENTS_FOR_RERANK", "12") or "12")

    app_password = _secret("APP_PASSWORD", "") or ""

    with st.sidebar:
        st.subheader("Settings")
        st.text_input("Pinecone index", value=index_name, disabled=True)
        st.text_input("Namespace", value=namespace, disabled=True)
        st.text_input("Parent store", value=parent_path, disabled=True)
        st.number_input("Pinecone top_k", value=pinecone_top_k, disabled=True, help="Set PINECONE_TOP_K in env to change")
        st.number_input("Rerank pool size", value=rerank_top_n, disabled=True, help="Set RERANK_TOP_N in env to change")
        st.number_input("Chunks to LLM", value=llm_chunks, disabled=True, help="Set LLM_CONTEXT_CHUNKS in env to change")

        unlocked = True
        if app_password:
            entered = st.text_input("App password", type="password")
            unlocked = entered == app_password
            if not unlocked:
                st.warning("Enter the correct password to use the chat.")

    if app_password and not unlocked:
        st.stop()

    parent_store = load_parent_store(parent_path)
    if not parent_store:
        st.warning(
            f"No parent store loaded from `{parent_path}`. Answers will fall back to child snippets only "
            "unless you generate `parent_store.json` with `ingest.py`."
        )

    openai_client = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(index_name)
    cohere_client = cohere.Client(api_key=cohere_key)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(
                            f"- **{s.get('speaker','')}** — *{s.get('title','')}* ({s.get('year','')}, {s.get('season','')})  \n"
                            f"  [{s.get('url','')}]({s.get('url','')})"
                        )

    if prompt := st.chat_input("Ask about General Conference…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating…"):
                sources: list[dict[str, Any]] = []
                qvec = embed_question(openai_client, embed_model, prompt)
                matches = pinecone_child_search(index, namespace, qvec, pinecone_top_k)
                parent_ids = ordered_unique_parent_ids(matches, max_parents=max_parents_for_rerank)
                docs, doc_meta = build_parent_documents(parent_store, parent_ids, matches)

                ranked_indices: list[tuple[int, float]]
                try:
                    ranked = cohere_rerank(
                        cohere_client,
                        rerank_model,
                        prompt,
                        docs,
                        top_n=min(rerank_top_n, len(docs)),
                    )
                    ranked_indices = ranked
                except Exception as e:
                    st.warning(f"Cohere rerank failed ({e}); using Pinecone order.")
                    ranked_indices = [(i, 0.0) for i in range(len(docs))]

                chosen: list[tuple[str, dict[str, Any]]] = []
                seen_pick: set[int] = set()
                for idx, _score in ranked_indices:
                    if idx in seen_pick:
                        continue
                    seen_pick.add(idx)
                    if 0 <= idx < len(docs):
                        chosen.append((docs[idx], doc_meta[idx]))
                    if len(chosen) >= llm_chunks:
                        break

                if not chosen:
                    reply = "No relevant passages were retrieved. Try rephrasing your question."
                else:
                    contexts = [c[0] for c in chosen]
                    sources = [c[1] for c in chosen]
                    reply = answer_question(openai_client, chat_model, prompt, contexts, sources)

                st.markdown(reply)
                if sources:
                    with st.expander("Sources used"):
                        for s in sources:
                            st.markdown(
                                f"- **{s.get('speaker','')}** — *{s.get('title','')}* ({s.get('year','')}, {s.get('season','')})  \n"
                                f"  [{s.get('url','')}]({s.get('url','')})"
                            )

        st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})


if __name__ == "__main__":
    main()
