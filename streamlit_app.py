from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from intelligent_explanation_agent.agent import ExplainableRecommendationAgent
from intelligent_explanation_agent.settings import Settings


TASK_TITLES = {
    "knowledge": "知识点推荐",
    "course": "课程推荐",
    "consume": "消费类别推荐",
}


def _status_text(status: str) -> tuple[str, str]:
    if status == "ok":
        return "DeepSeek 已完成综合解释", "success"
    if status == "not_configured":
        return "未配置 DeepSeek API，当前仅展示核心推荐结果", "warning"
    if status == "unavailable":
        return "DeepSeek API 暂不可用，当前展示核心推荐结果", "warning"
    return f"解释状态：{status}", "info"


@st.cache_resource(show_spinner=False)
def get_agent() -> ExplainableRecommendationAgent:
    settings = Settings.from_env()
    recommender = settings.make_recommender()
    return ExplainableRecommendationAgent(
        recommender=recommender,
        llm=settings.make_llm(),
        recent_limit=settings.recent_limit,
        include_user_id_in_api=settings.include_user_id_in_api,
    )


@st.cache_data(show_spinner=False)
def get_user_ids() -> list[int]:
    return get_agent().recommender.list_user_ids()


def _safe_score(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def _summary_value(summary: dict[str, Any], *names: str, default: Any = 0) -> Any:
    for name in names:
        if name in summary:
            return summary[name]
    return default


def _format_text_sequence(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return "；".join(str(item) for item in value if str(item).strip())
    return str(value)


def render_recommendation_table(task: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        st.info(f"暂无{TASK_TITLES.get(task, task)}结果")
        return

    table = pd.DataFrame(
        [
            {
                "排名": index,
                "推荐项": row.get("item", ""),
                "得分": _safe_score(row.get("score")),
            }
            for index, row in enumerate(rows, start=1)
        ]
    )
    st.dataframe(table, hide_index=True, use_container_width=True)


def render_snapshot(snapshot: dict[str, Any]) -> None:
    st.subheader("近期三模态用户快照")
    summary = snapshot.get("summary", {})

    learning_count = _summary_value(
        summary, "learning_count", "learning_events", default=len(snapshot.get("learning", []))
    )
    consumption_count = _summary_value(
        summary, "consumption_count", "consumption_events", default=len(snapshot.get("consumption", []))
    )
    access_count = _summary_value(
        summary, "access_count", "access_events", default=len(snapshot.get("access", []))
    )
    accuracy = _summary_value(summary, "recent_learning_accuracy", "learning_accuracy", default=None)

    cols = st.columns(4)
    cols[0].metric("学习事件", learning_count)
    cols[1].metric("消费事件", consumption_count)
    cols[2].metric("门禁事件", access_count)
    cols[3].metric("近期正确率", "-" if accuracy is None else f"{float(accuracy):.2%}")

    tabs = st.tabs(["学习模态", "消费模态", "门禁模态"])
    for key, tab in zip(("learning", "consumption", "access"), tabs):
        with tab:
            rows = snapshot.get(key, [])
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            else:
                st.info("暂无近期记录")


def render_global_explanation(explanation: dict[str, Any] | None) -> None:
    st.subheader("智能体综合解释")
    if not explanation:
        st.info(
            "当前没有可用的 DeepSeek 综合解释。配置 DEEPSEEK_API_KEY 后，"
            "系统会结合近期三模态快照和推荐结果生成说明。"
        )
        return

    fields = [
        ("user_summary", "用户近期行为摘要"),
        ("overall_reason", "综合推荐理由"),
        ("learning_reason", "学习/知识点推荐依据"),
        ("course_reason", "课程推荐依据"),
        ("consumption_reason", "消费推荐依据"),
        ("acceptance_note", "用户端接受理由"),
    ]
    for key, title in fields:
        value = explanation.get(key)
        if value:
            st.markdown(f"**{title}**")
            st.write(value)

    modality = explanation.get("modality_insights")
    if isinstance(modality, dict) and modality:
        with st.expander("三模态证据概览", expanded=False):
            for key, title in [
                ("learning", "学习模态"),
                ("consumption", "消费模态"),
                ("access", "门禁模态"),
            ]:
                if modality.get(key):
                    st.markdown(f"**{title}**")
                    st.write(modality[key])

    caveat_text = _format_text_sequence(explanation.get("caveats"))
    if caveat_text:
        st.caption("说明边界：" + caveat_text)


def choose_user_id(user_ids: list[int]) -> int | None:
    st.sidebar.header("学生选择")
    mode = st.sidebar.radio("选择方式", ["下拉选择", "手动输入"], horizontal=True)
    if mode == "下拉选择":
        if not user_ids:
            st.sidebar.warning("没有可选学生 ID")
            return None
        return int(st.sidebar.selectbox("学生 ID", user_ids))

    text = st.sidebar.text_input("输入学生 ID", value=str(user_ids[0]) if user_ids else "")
    if not text.strip():
        return None
    try:
        return int(text)
    except ValueError:
        st.sidebar.error("学生 ID 必须是整数")
        return None


def main() -> None:
    st.set_page_config(page_title="智能体可解释推荐系统", page_icon="🎓", layout="wide")
    st.title("🎓 智能体可解释推荐系统")
    st.caption("输入或选择学生 ID，系统调用核心推荐网络生成结果，并由上层智能体组织综合解释。")

    try:
        with st.spinner("正在加载用户列表与推荐模型，首次启动可能需要一些时间..."):
            user_ids = get_user_ids()
    except Exception as exc:  # pragma: no cover - UI boundary
        st.error(f"模型或数据加载失败：{exc}")
        return

    user_id = choose_user_id(user_ids)
    require_explanation = st.sidebar.checkbox("要求必须生成 DeepSeek 解释", value=False)
    run = st.sidebar.button("生成推荐与解释", type="primary", use_container_width=True)

    st.sidebar.divider()
    st.sidebar.caption("DeepSeek 配置统一写在 intelligent_explanation_agent/config.local.json。")

    if user_id is None:
        st.info("请先选择或输入学生 ID。")
        return
    if not run:
        st.info("点击左侧按钮后生成该学生的推荐结果。")
        return

    try:
        with st.spinner(f"正在为学生 {user_id} 生成推荐与综合解释..."):
            result = get_agent().run(user_id, require_explanation=require_explanation)
    except Exception as exc:  # pragma: no cover - UI boundary
        st.error(f"生成失败：{exc}")
        return

    status_message, status_level = _status_text(result.get("explanation_status", "unknown"))
    getattr(st, status_level)(status_message)
    if result.get("explanation_error"):
        st.caption(result["explanation_error"])

    top_cols = st.columns([1, 3])
    top_cols[0].metric("学生 ID", result.get("user_id", user_id))
    top_cols[1].write("推荐结果来自核心推荐网络；综合解释来自上层智能体/DeepSeek，不改变推荐排序。")

    render_global_explanation(result.get("explanation"))

    st.subheader("推荐结果")
    recommendations = result.get("recommendations", {})
    rec_tabs = st.tabs([TASK_TITLES[task] for task in TASK_TITLES])
    for task, tab in zip(TASK_TITLES, rec_tabs):
        with tab:
            render_recommendation_table(task, recommendations.get(task, []))

    render_snapshot(result.get("snapshot", {}))


def _run_with_streamlit_when_called_by_python() -> bool:
    """Let `python streamlit_app.py` behave like `streamlit run streamlit_app.py`."""
    if get_script_run_ctx() is not None:
        return False
    if os.getenv("STREAMLIT_APP_BOOTSTRAPPED") == "1":
        return False
    os.environ["STREAMLIT_APP_BOOTSTRAPPED"] = "1"
    command = [sys.executable, "-m", "streamlit", "run", __file__]
    subprocess.run(command, check=False)
    return True


if __name__ == "__main__":
    if not _run_with_streamlit_when_called_by_python():
        main()
