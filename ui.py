"""Streamlit dashboard for Rail-CRAG."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.config import load_settings
from src.graph.builder import build_crag_graph

st.set_page_config(page_title="Rail-CRAG Agent", layout="wide")

st.title("🚄 Rail-CRAG: 铁路标准智能问答")
st.markdown("基于 Corrective Retrieval Augmented Generation (CRAG) 的交互式可视化")

settings = load_settings(require_keys=False)

with st.sidebar:
    st.header("⚙️ 参数配置")
    upper_threshold = st.slider("Correct 阈值", -1.0, 1.0, settings.upper_threshold, 0.1)
    lower_threshold = st.slider("Incorrect 阈值", -1.0, 1.0, settings.lower_threshold, 0.1)
    st.info(
        f"当前策略:\n- 分数 > {upper_threshold}: 🟢 Correct\n"
        f"- 分数 < {lower_threshold}: 🔴 Incorrect\n"
        f"- 中间: 🟡 Ambiguous"
    )

os.environ["CRAG_UPPER_THRESHOLD"] = str(upper_threshold)
os.environ["CRAG_LOWER_THRESHOLD"] = str(lower_threshold)

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
        message_placeholder = st.empty()
        full_response = ""

        with st.status("🧠 CRAG 思考中...", expanded=True) as status:
            graph = build_crag_graph()
            inputs: Dict[str, Any] = {"question": prompt}
            step_container = st.container()

            try:
                for output in graph.stream(inputs):
                    for node_name, state in output.items():
                        if node_name == "retrieve" and "retrieved_documents" in state:
                            docs: List[dict] = state["retrieved_documents"]
                            status.write(f"🔍 检索到 {len(docs)} 条相关文档")
                            with step_container.expander("查看原始检索 (Raw Retrieval)"):
                                for doc in docs:
                                    preview = (doc.get("content", "") or "")[:200]
                                    st.text(f"ID: {doc.get('id')}\nContent: {preview}...")

                        if "evaluation_scores" in state:
                            scores = state.get("evaluation_scores", {})
                            confidence = state.get("confidence", "unknown")
                            color = "green" if confidence == "correct" else "red" if confidence == "incorrect" else "orange"
                            status.markdown(f"⚖️ 评估结果: :{color}[**{confidence.upper()}**]")
                            df = pd.DataFrame(list(scores.items()), columns=["DocID", "Score"])
                            step_container.table(df)

                        if "knowledge_strips" in state and state["knowledge_strips"]:
                            strips = state["knowledge_strips"]
                            status.write(f"🧪 知识提炼完成 (Extracted {len(strips)} strips)")

                        if "search_results" in state and state["search_results"]:
                            results = state["search_results"]
                            status.write(f"🌐 触发联网搜索 (Found {len(results)} results)")
                            with step_container.expander("搜索结果内容"):
                                for res in results:
                                    st.markdown(f"- {res[:200]}...")

                        if "final_answer" in state:
                            full_response = state["final_answer"]

                status.update(label="✅ 回答生成完毕", state="complete", expanded=False)
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as exc:
                status.update(label="❌ 发生错误", state="error")
                st.error(f"Error: {exc}")
