from __future__ import annotations

from typing import Any

import streamlit as st


def render_sources_expander(sources: list[dict[str, Any]], title: str = "Sources") -> None:
    with st.expander(title):
        for source in sources:
            st.markdown(
                f"- **{source.get('speaker','')}** — *{source.get('title','')}* "
                f"({source.get('year','')}, {source.get('season','')})  \n"
                f"  [{source.get('url','')}]({source.get('url','')})"
            )


def render_chat_history(messages: list[dict[str, Any]]) -> None:
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources_expander(msg["sources"], title="Sources")
