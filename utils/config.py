from __future__ import annotations

import os
from dataclasses import dataclass

import streamlit as st


def secret(key: str, default: str | None = None) -> str | None:
    """Read from environment first, then Streamlit Cloud secrets."""
    env_value = os.getenv(key)
    if env_value:
        return env_value
    try:
        return str(st.secrets[key])
    except Exception:
        return default


@dataclass(frozen=True)
class AppConfig:
    openai_key: str
    pinecone_key: str
    cohere_key: str
    index_name: str
    namespace: str
    parent_path: str
    embed_model: str
    chat_model: str
    rerank_model: str
    pinecone_top_k: int
    rerank_top_n: int
    llm_chunks: int
    max_parents_for_rerank: int
    app_password: str


def load_app_config() -> AppConfig:
    return AppConfig(
        openai_key=secret("OPENAI_API_KEY") or "",
        pinecone_key=secret("PINECONE_API_KEY") or "",
        cohere_key=secret("COHERE_API_KEY") or "",
        index_name=secret("PINECONE_INDEX", "conference-rag") or "conference-rag",
        namespace=secret("PINECONE_NAMESPACE", "talks-v1") or "talks-v1",
        parent_path=secret("PARENT_STORE_PATH", "storage/parent_store.json") or "storage/parent_store.json",
        embed_model=secret("EMBEDDING_MODEL", "text-embedding-3-small") or "text-embedding-3-small",
        chat_model=secret("CHAT_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
        rerank_model=secret("COHERE_RERANK_MODEL", "rerank-english-v3.0") or "rerank-english-v3.0",
        pinecone_top_k=int(secret("PINECONE_TOP_K", "20") or "20"),
        rerank_top_n=int(secret("RERANK_TOP_N", "8") or "8"),
        llm_chunks=int(secret("LLM_CONTEXT_CHUNKS", "5") or "5"),
        max_parents_for_rerank=int(secret("MAX_PARENTS_FOR_RERANK", "12") or "12"),
        app_password=secret("APP_PASSWORD", "") or "",
    )
