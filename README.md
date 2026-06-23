# Smart Campus Multi-Source Fusion Deep Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**智慧校园多源融合深度推荐系统** — 融合MOOC学习行为、一卡通消费与门禁记录的下一代智能推荐引擎

[English](README.md) | [中文](README_CN.md)

</div>

---

## 项目简介

本项目实现了一个面向智慧校园场景的**多源融合深度推荐系统**，创新性地将MOOC学习数据、一卡通消费记录与门禁日志进行跨域统一建模，系统性地解决了多模态数据融合、用户兴趣漂移、冷启动推荐等核心挑战。

### 核心特性

| 特性 | 描述 |
|------|------|
| 🔗 **跨域用户统一** | 基于同学科约束下的模拟配对策略，实现MOOC用户与校园一卡通/门禁记录的身份关联 |
| 🧠 **多兴趣表征** | 采用DIEN+AUGRU双塔架构，分别捕捉用户的长期稳定偏好与短期动态兴趣 |
| 📚 **知识感知推荐** | 融合MOOCCubeX知识点图谱，引入TransE知识嵌入与先修课程关系 |
| ⚡ **多任务学习** | PLE(多任务利诱网络)联合优化课程推荐、消费预测与门禁行为建模 |
| 🎯 **冷启动支持** | 基于用户画像(学科/性别/年级)的零行为新用户推理能力 |
| 📊 **多样性保障** | MMR重排序 + LogQ热门校正 + 多样性配额机制 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        输入层 (多源数据)                          │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   MOOC 学习行为  │   一卡通消费记录  │       门禁日志              │
│  (MOOCCubeX)    │   (Teddy数据)    │      (Teddy数据)            │
├─────────────────┴─────────────────┴─────────────────────────────┤
│                      统一ID映射层                                │
│              (global_user_id 跨域关联)                          │
├─────────────────────────────────────────────────────────────────┤
│                      特征编码层                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  DIEN (学习)  │  │  DIEN (消费)  │  │  DIEN (门禁)  │          │
│  │  + AUGRU     │  │  + 多兴趣向量  │  │  + 空间活动   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
├─────────────────────────────────────────────────────────────────┤
│                      兴趣漂移模块                                 │
│        (长期兴趣 ◄── 动态门控 ◄── 短期兴趣)                       │
├─────────────────────────────────────────────────────────────────┤
│                      双塔召回层                                   │
│                 (Top-100 候选生成)                               │
├─────────────────────────────────────────────────────────────────┤
│                      PLE精排层                                    │
│          (课程推荐 / 消费预测 / 门禁行为 多任务)                  │
├─────────────────────────────────────────────────────────────────┤
│                      MMR重排序                                   │
│              (多样性保障 + 热门度校正)                           │
├─────────────────────────────────────────────────────────────────┤
│                      Top-K 输出                                 │
│              (个性化课程推荐列表)                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 目录结构

```
recommender_system/
├── MOOCCubeX/                    # 清华大学MOOCCubeX数据集
│   └── MOOCCubeX-main/
│       ├── entities/             # 实体数据 (用户/课程/知识点/习题)
│       ├── relations/           # 关系数据 (用户-习题/知识点-课程等)
│       ├── prerequisites/        # 先修课程知识图谱
│       ├── scripts/              # 数据集处理工具脚本
│       └── docs/                 # 数据集文档
│
├── smartcampus_recommender/      # 核心推荐系统代码
│   ├── config.py                 # 统一超参数配置
│   ├── main.py                   # 训练/验证/测试统一入口
│   ├── main_enhanced.py          # 增强版主程序
│   ├── infer.py                  # 单用户推理脚本
│   ├── cold_start_infer.py       # 冷启动推理
│   ├── data/                     # 数据加载与预处理
│   │   ├── dataset.py            # PyTorch Dataset实现
│   │   ├── preprocessing.py      # 特征工程
│   │   ├── knowledge_mapping.py  # 知识点图谱映射
│   │   └── vocab.py              # 词表与ID映射
│   ├── models/                   # 深度学习模型库
│   │   ├── dien.py               # DIEN (深度兴趣演化网络)
│   │   ├── akt.py                 # AKT (知识追踪增强注意力)
│   │   ├── ple.py                 # PLE (多任务利诱网络)
│   │   ├── two_tower.py          # 双塔召回模型
│   │   ├── transe.py              # TransE知识嵌入
│   │   └── full_model.py         # 完整推荐模型
│   ├── training/                  # 训练流程
│   │   ├── trainer.py            # 训练器与早停策略
│   │   ├── losses.py             # 损失函数定义
│   │   └── metrics.py            # 评估指标 (Hit@K, MRR, NDCG)
│   ├── inference/                # 推理服务
│   │   └── recommender.py        # Top-K推荐解码
│   ├── visualization/            # 可视化模块
│   │   └── plots.py              # 论文图表生成
│   ├── reporting/               # 报告生成
│   │   ├── excel_report.py      # Excel实验报告
│   │   └── thesis_text.py       # 论文文字素材
│   └── utils/                    # 工具函数
│
├── my_output/                    # 预处理后的数据输出
├── result/                       # 实验结果输出
├── logs/                         # 训练日志
└── cache/                        # 缓存文件
```

---

## 快速开始

### 环境依赖

```bash
pip install -r smartcampus_recommender/requirements.txt
```

核心依赖：
- `torch >= 2.0`
- `numpy >= 1.23`
- `pandas >= 1.5`
- `scikit-learn >= 1.2`
- `matplotlib >= 3.6`
- `seaborn >= 0.12`
- `openpyxl >= 3.1`
- `tqdm >= 4.65`

### 数据准备

本项目使用两套数据源：

1. **MOOCCubeX数据集** (清华大学) — 包含MOOC课程、学习行为记录
2. **泰迪杯竞赛数据** — 包含一卡通消费记录与门禁日志

**完整数据处理流程：**

```powershell
# 1. 克隆项目
git clone https://github.com/your-repo/recommender_system.git
cd recommender_system

# 2. 放置原始数据到对应目录
# - MOOCCubeX数据 → MOOCCubeX/MOOCCubeX-main/
# - 泰迪杯数据   → dd0f7-main/

# 3. 运行数据对齐脚本
python build_cross_dataset.py

# 4. 生成的数据输出到 my_output/
```

### 模型训练

```powershell
cd smartcampus_recommender

# 标准训练
python main.py --data-root ../my_output

# 增强版 (含Time2Vec、相对位置编码)
python main_enhanced.py

# 快速验证 (不运行消融实验)
python main.py --data-root ../my_output --quick
```

### 模型推理

```powershell
# 对指定用户推理
python infer.py --data-root ../my_output --user-id 1

# 冷启动用户推荐
python cold_start_infer.py --subject 计算机类 --gender 男 --grade 2019
```

### 输出结果

```
result/
├── best_model.pth              # 最优模型权重
├── metric_result.xlsx          # Excel格式实验报告
├── test_metrics.json           # 测试指标JSON
├── ablation_metrics.json       # 消融实验结果
├── test_topk_recommendations.csv  # Top-K推荐结果
├── thesis_ready_text.md        # 论文可用文字素材
└── hierarchy_mapping_stats.json # 知识图谱统计

logs/
└── training.log                # 完整训练日志
```

---

## 核心技术模块

### 1. 多源数据融合

| 数据源 | 字段 | 建模方式 |
|--------|------|----------|
| MOOC学习 | 问题ID、时间戳、正确性 | DIEN + 知识点嵌入 |
| 一卡通消费 | 消费类型、金额、时间 | DIEN + 多兴趣向量 |
| 门禁记录 | 门禁点、时间、刷卡类型 | DIEN + 空间活动模式 |

### 2. 知识增强模块

- **TransE嵌入**：将知识点与课程映射到统一向量空间
- **先修关系**：利用MOOCCubeX的课程先修图谱进行知识追踪
- **知识点覆盖**：记录推荐课程的知识点覆盖度

### 3. 兴趣演化建模

```
短期兴趣 (AUGRU) ◄── 动态门控权重 ──► 长期兴趣 (DIEN)
      │                                        │
      └────────── 兴趣漂移向量 (Interest Drift) ──┘
```

### 4. 多任务学习 (PLE)

```
          Shared Experts
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
┌───────┐ ┌───────┐ ┌───────┐
│课程推荐│ │消费预测│ │门禁行为│
│ Expert│ │ Expert│ │ Expert│
└───────┘ └───────┘ └───────┘
```

---

## 实验配置

核心超参数可通过 `config.py` 修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embedding_dim` | 64 | 嵌入维度 |
| `hidden_dim` | 128 | 隐藏层维度 |
| `num_interests` | 4 | 多兴趣向量数量 |
| `retrieval_topk` | 100 | 召回候选数量 |
| `learning_rate` | 1e-3 | 学习率 |
| `batch_size` | 64 | 批大小 |
| `epochs` | 60 | 训练轮数 |

---

## 数据兼容性

本系统兼容以下文件名与字段格式：

**文件名映射：**
- `mooc_user_mapped.jsonl` ↔ `sampled_mooc_user.json`
- `user_problem_mapped.jsonl` ↔ `sampled_user_problem.json`
- `teddy_consume_mapped.csv` ↔ `mapped_teddy_consume.csv`

**字段名映射：**
- `correct` / `is_correct`
- `time` / `submit_time` / `Date`
- `consume_type` / `Dept` / `Type`

---

## 相关项目

| 项目 | 说明 |
|------|------|
| [MOOCCubeX](https://github.com/thsd/MOOCCubeX) | 清华大学MOOC数据集 |
| [MOOCCube](https://github.com/thsd/MOOCCube) | MOOCCube原始版本 |
| [Concept-Acquisition-Pipeline](https://github.com/yujifan0326/Concept-Acquisition-Pipeline) | 知识点提取流水线 |
| [prerequisite-prediction-co-training](https://github.com/luogan1234/prerequisite-prediction-co-training) | 先修课程预测 |

---

## 致谢

- 数据集来源：[MOOCCubeX](https://lfs.aminer.cn/misc/moocdata/publications/mooccubex.pdf) - 清华大学知识工程实验室
- 平台支持：[学堂在线](https://www.xuetangx.com/) - 中国最大MOOC平台之一

---

## 许可证

本项目仅供学术研究使用。MOOCCubeX数据集遵循其原始许可证约定。

---

<div align="center">

**Star ⭐ 如果这个项目对你有帮助！**

</div>
