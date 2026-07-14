# Streamlit 前端说明

这个前端用于用户端演示：输入或选择学生 ID 后，系统调用上层可解释智能体，返回核心推荐结果和 DeepSeek 生成的综合解释。

## 安装依赖

```powershell
pip install -r intelligent_explanation_agent/requirements-streamlit.txt
```

核心推荐器仍然需要原项目依赖，例如 `torch`：

```powershell
pip install -r smartcampus_recommender/requirements.txt
```

## 统一配置文件

推荐直接修改：

```text
intelligent_explanation_agent/config.local.json
```

主要配置项：

```json
{
  "project_root": ".",
  "data_root": "my_output",
  "checkpoint_path": "best_model.pth",
  "recommend_topk": 10,
  "recent_events_per_modality": 10,
  "include_user_id_in_api": false,
  "glm": {
    "api_key": "你的 DeepSeek API Key",
    "api_base": "https://api.deepseek.com",
    "model": "deepseek-v4-flash",
    "timeout_seconds": 60
  }
}
```

说明：为了兼容旧代码，配置块名称仍保留为 `glm`，但里面现在填写 DeepSeek 的接口信息即可。

`config.local.json` 已加入 `.gitignore`，适合存放真实密钥。如果需要换一个配置文件，可以设置：

```powershell
$env:INTELLIGENT_AGENT_CONFIG="F:\projects\recommender_system\intelligent_explanation_agent\config.local.json"
```

环境变量优先级高于配置文件。

## DeepSeek 环境变量配置

当前项目默认使用 DeepSeek 的 OpenAI-compatible Chat Completions 接口：

```powershell
$env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"
$env:DEEPSEEK_API_BASE="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-v4-flash"
$env:DEEPSEEK_TIMEOUT_SECONDS="60"
```

系统也兼容旧的 `GLM_*` 和 `EXPLAIN_*` 变量名，但建议新配置统一使用 `DEEPSEEK_*`。

## 启动

```powershell
streamlit run streamlit_app.py
```

## 页面功能

- 下拉选择学生 ID；
- 手动输入学生 ID；
- 展示知识点、课程、消费类别三类推荐结果；
- 展示上层智能体调用 DeepSeek 生成的用户端综合解释；
- 展示近期学习、消费、门禁三模态行为快照。

未配置 `DEEPSEEK_API_KEY` 时，前端仍会展示核心推荐结果，但解释区域会提示解释 API 未配置。
