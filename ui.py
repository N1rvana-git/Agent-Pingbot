"""Streamlit demo for Rail-CRAG."""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.config import load_settings
from src.graph.builder import build_crag_graph
from src.graph.nodes import CRAGNodes

st.set_page_config(page_title="Rail-CRAG Agent", layout="wide")

st.title("🚄 Rail-CRAG: 铁路标准智能问答")
st.caption("CRAG（Corrective RAG）+ 混合检索（BM25 + 向量 + RRF + 重排）")

settings = load_settings(require_keys=False)

with st.sidebar:
    st.header("⚙️ 检索配置")
    mode = st.selectbox(
        "检索模式",
        options=["hybrid", "bm25", "vector"],
        index=["hybrid", "bm25", "vector"].index(settings.retrieval_mode),
        help="hybrid = BM25 与向量检索经 RRF 融合",
    )
    rerank = st.checkbox("启用重排（Rerank）", value=settings.rerank)
    top_k = st.slider("Top-K", 1, 10, settings.retriever_k)
    st.divider()
    st.caption(
        "确定性话术与检索结果都会标注来源章节；未配置模型时直接返回检索上下文。"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("请输入关于铁路标准的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.status("🧠 CRAG 检索与生成中...", expanded=True) as status:
            runtime = dataclasses.replace(
                settings,
                retrieval_mode=mode,
                rerank=rerank,
                retriever_k=top_k,
            )
            nodes = CRAGNodes(runtime)
            graph = build_crag_graph(nodes)
            try:
                result: Dict[str, Any] = graph.invoke({"question": prompt})
                documents: List[dict] = result.get("retrieved_documents", [])
                trace = result.get("retrieval_trace", {})
                confidence = str(result.get("confidence", "unknown")).upper()

                status.write(
                    f"🔍 {trace.get('mode')} 检索到 {len(documents)} 条片段（rerank={trace.get('rerank')}）"
                )
                if documents:
                    frame = pd.DataFrame(
                        [
                            {
                                "score": doc.get("score"),
                                "bm25_rank": doc.get("bm25_rank"),
                                "vector_rank": doc.get("vector_rank"),
                                "path": doc.get("path"),
                                "source": doc.get("source"),
                            }
                            for doc in documents
                        ]
                    )
                    st.dataframe(frame, use_container_width=True, hide_index=True)
                if trace.get("vector_error"):
                    st.warning(f"向量检索降级：{trace['vector_error']}")

                color = {"CORRECT": "green", "INCORRECT": "red"}.get(confidence, "orange")
                status.markdown(f"⚖️ 评估结果: :{color}[**{confidence}**]")

                if result.get("knowledge_strips"):
                    status.write(f"🧪 知识提炼：{len(result['knowledge_strips'])} 段")
                if result.get("search_results"):
                    status.write(f"🌐 联网补充：{len(result['search_results'])} 条")

                answer = result.get("final_answer", "")
                status.update(label="✅ 回答生成完毕", state="complete", expanded=False)
                placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as exc:  # noqa: BLE001 - surface the failure in the UI
                status.update(label="❌ 发生错误", state="error")
                st.error(f"Error: {exc}")
