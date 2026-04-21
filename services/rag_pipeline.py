from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cohere
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone

from utils.config import AppConfig


@st.cache_resource
def load_parent_store(path: str) -> dict[str, Any]:
    parent_path = Path(path)
    if not parent_path.exists():
        return {}
    with parent_path.open(encoding="utf-8") as handle:
        return json.load(handle)


class RagService:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._openai_client = OpenAI(api_key=config.openai_key)
        self._pinecone_index = Pinecone(api_key=config.pinecone_key).Index(config.index_name)
        self._cohere_client = cohere.Client(api_key=config.cohere_key)

    def _embed_question(self, text: str) -> list[float]:
        response = self._openai_client.embeddings.create(model=self._config.embed_model, input=[text])
        return response.data[0].embedding

    def _pinecone_child_search(self, vector: list[float], top_k: int) -> list[dict[str, Any]]:
        response = self._pinecone_index.query(
            vector=vector,
            top_k=top_k,
            namespace=self._config.namespace,
            include_metadata=True,
        )
        matches = getattr(response, "matches", None) or response.get("matches", [])  # type: ignore[union-attr]
        output: list[dict[str, Any]] = []
        for match in matches:
            metadata = getattr(match, "metadata", None) or match.get("metadata") or {}
            match_id = getattr(match, "id", None) or match.get("id")
            score = getattr(match, "score", None) if getattr(match, "score", None) is not None else match.get("score")
            output.append({"id": match_id, "score": score, "metadata": dict(metadata) if metadata else {}})
        return output

    def _ordered_unique_parent_ids(self, matches: list[dict[str, Any]], max_parents: int) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for match in matches:
            parent_id = (match.get("metadata") or {}).get("parent_id")
            if not parent_id or not isinstance(parent_id, str) or parent_id in seen:
                continue
            seen.add(parent_id)
            ordered.append(parent_id)
            if len(ordered) >= max_parents:
                break
        return ordered

    def _build_parent_documents(
        self,
        parent_store: dict[str, Any],
        parent_ids: list[str],
        matches: list[dict[str, Any]],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        best_child_meta: dict[str, dict[str, Any]] = {}
        for match in matches:
            meta = match.get("metadata") or {}
            parent_id = meta.get("parent_id")
            if parent_id and parent_id not in best_child_meta:
                best_child_meta[str(parent_id)] = meta

        documents: list[str] = []
        metadata_list: list[dict[str, Any]] = []
        for parent_id in parent_ids:
            record = parent_store.get(parent_id)
            if record and record.get("text"):
                documents.append(str(record["text"]))
                metadata_list.append(
                    {
                        "parent_id": parent_id,
                        "title": record.get("title", ""),
                        "speaker": record.get("speaker", ""),
                        "year": record.get("year", ""),
                        "season": record.get("season", ""),
                        "url": record.get("url", ""),
                    }
                )
                continue

            meta = best_child_meta.get(parent_id, {})
            title = meta.get("title", "")
            speaker = meta.get("speaker", "")
            year = meta.get("year", "")
            season = meta.get("season", "")
            snippet = meta.get("text", "")
            fallback = f"[Speaker: {speaker}, Year: {year}, Season: {season}, Title: {title}]\n{snippet}"
            documents.append(fallback)
            metadata_list.append(
                {
                    "parent_id": parent_id,
                    "title": title,
                    "speaker": speaker,
                    "year": year,
                    "season": season,
                    "url": meta.get("url", ""),
                    "note": "parent not in local store; using child snippet from Pinecone metadata",
                }
            )
        return documents, metadata_list

    def _cohere_rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        if not documents:
            return []
        response = self._cohere_client.rerank(
            model=self._config.rerank_model,
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
        )
        results = getattr(response, "results", None) or []
        output: list[tuple[int, float]] = []
        for result in results:
            output.append((int(getattr(result, "index", 0)), float(getattr(result, "relevance_score", 0.0))))
        return output

    def _answer_question(self, user_question: str, contexts: list[str], sources: list[dict[str, Any]]) -> str:
        joined_sources = []
        for i, (context, source) in enumerate(zip(contexts, sources), start=1):
            cite = (
                f"{source.get('speaker','')} — {source.get('title','')} "
                f"({source.get('year','')}, {source.get('season','')})"
            )
            joined_sources.append(f"### Source {i}\n{cite}\nURL: {source.get('url','')}\n\n{context}")

        system_prompt = (
            "You are a careful research assistant for General Conference talks. "
            "Answer ONLY using the provided sources. If the sources do not contain enough "
            "information, say you are not sure. For factual claims, cite the speaker and year "
            "in plain text (no fake quotes). Do not invent talks or quotes."
        )
        user_prompt = f"Question:\n{user_question}\n\nSources:\n" + "\n\n".join(joined_sources)
        completion = self._openai_client.chat.completions.create(
            model=self._config.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return (completion.choices[0].message.content or "").strip()

    def search_talks(self, prompt: str, top_k: int = 100) -> list[dict[str, Any]]:
        query_vector = self._embed_question(prompt)
        matches = self._pinecone_child_search(query_vector, top_k=top_k)

        # Collapse child chunk hits into unique talks, keeping the highest child score per talk.
        talks_by_parent: dict[str, dict[str, Any]] = {}
        for match in matches:
            metadata = match.get("metadata") or {}
            parent_id = metadata.get("parent_id")
            if not parent_id:
                continue

            score_value = match.get("score")
            try:
                score = float(score_value) if score_value is not None else 0.0
            except (TypeError, ValueError):
                score = 0.0

            parent_key = str(parent_id)
            existing = talks_by_parent.get(parent_key)
            if existing and existing.get("score", 0.0) >= score:
                continue

            talks_by_parent[parent_key] = {
                "parent_id": parent_key,
                "score": score,
                "title": metadata.get("title", ""),
                "speaker": metadata.get("speaker", ""),
                "year": metadata.get("year", ""),
                "season": metadata.get("season", ""),
                "url": metadata.get("url", ""),
            }

        return sorted(talks_by_parent.values(), key=lambda item: item.get("score", 0.0), reverse=True)

    def run_query(self, prompt: str, parent_store: dict[str, Any]) -> tuple[str, list[dict[str, Any]], str | None]:
        query_vector = self._embed_question(prompt)
        matches = self._pinecone_child_search(query_vector, self._config.pinecone_top_k)
        parent_ids = self._ordered_unique_parent_ids(matches, max_parents=self._config.max_parents_for_rerank)
        documents, document_meta = self._build_parent_documents(parent_store, parent_ids, matches)

        warning_message: str | None = None
        try:
            ranked_indices = self._cohere_rerank(prompt, documents, top_n=min(self._config.rerank_top_n, len(documents)))
        except Exception as exc:
            warning_message = f"Cohere rerank failed ({exc}); using Pinecone order."
            ranked_indices = [(idx, 0.0) for idx in range(len(documents))]

        chosen: list[tuple[str, dict[str, Any]]] = []
        seen_indices: set[int] = set()
        for idx, _score in ranked_indices:
            if idx in seen_indices:
                continue
            seen_indices.add(idx)
            if 0 <= idx < len(documents):
                chosen.append((documents[idx], document_meta[idx]))
            if len(chosen) >= self._config.llm_chunks:
                break

        if not chosen:
            return "No relevant passages were retrieved. Try rephrasing your question.", [], warning_message

        contexts = [item[0] for item in chosen]
        sources = [item[1] for item in chosen]
        reply = self._answer_question(prompt, contexts, sources)
        return reply, sources, warning_message
