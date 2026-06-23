# 智慧校园多源融合深度推荐系统

## 目录

```text
smartcampus_recommender/
├── config.py                 # 全部超参数
├── main.py                   # 训练、验证、测试、消融、绘图、报告统一入口
├── infer.py                  # 任意用户独立推理
├── data/                     # 数据加载、特征工程、Dataset、padding/mask
├── models/                   # TransE、AKT、DIEN、映射网络、PLE及完整模型
├── training/                 # 联合损失、指标、训练器、早停和消融
├── inference/                # Top-K解码与测试集结果导出
├── visualization/            # 六类论文图片
├── reporting/                # Excel与论文文字
└── utils/                    # 日志、随机种子、路径兼容
```

## 运行

在本目录执行：

```powershell
python main.py --data-root ../my_output
```

增强版可直接运行（默认读取 `../my_output`）：

```powershell
python main_enhanced.py
```

增强版包含课程链路知识点回退、Time2Vec、相对位置、时间衰减注意力，以及
小时/星期/节假日Embedding。首次运行会生成新的v3缓存和
`result/knowledge_mapping_stats.json`。

首次建议运行端到端快速检查：

```powershell
python main.py --data-root ../my_output --quick
```

正式实验不要使用 `--quick` 或 `--skip-ablations`。完整运行自动生成：

```text
best_model.pth
fig/*.png
result/metric_result.xlsx
result/test_metrics.json
result/ablation_metrics.json
result/test_topk_recommendations.csv
result/thesis_ready_text.md
logs/training.log
```

对指定用户推理：

```powershell
python infer.py --data-root ../my_output --user-id 1
```

## 数据兼容

代码同时识别需求中的文件名和本项目实际输出名，例如：

- `mooc_user_mapped.jsonl` / `sampled_mooc_user.json`
- `user_problem_mapped.jsonl` / `sampled_user_problem.json`
- `teddy_consume_mapped.csv` / `mapped_teddy_consume.csv`

同样兼容 `correct/is_correct`、`time/submit_time/Date`、`consume_type/Dept/Type` 等字段。

## 统一ID建模假设

模型按实验要求，将共享 `global_user_id` 的MOOC学习记录与一卡通消费、门禁记录
直接视为同一学生。跨域映射采用逐用户 `z_behavior` 与该用户DIEN表征 `z_real`
的一一对齐损失，不再使用同学科用户均值。由于统一ID由原始独立数据合成，论文局限性
仍会自动说明这一实验假设。

推荐链路为：`用户多源表征 → 双塔Top-N召回 → PLE多任务精排 → Top-K结果`。
双塔候选数、温度系数及召回损失权重分别由 `retrieval_topk`、
`retrieval_temperature`、`retrieval_weight` 配置。

兴趣漂移模块采用长期DIEN、短期AUGRU、动态门控和多兴趣向量；主要配置为
`short_interest_window`、`num_interests` 与 `drift_smoothing_weight`，并自动生成
`fig/07_interest_drift.png`。

消费与门禁使用两套独立DIEN/AUGRU编码器。消费分支学习品类偏好并提供多兴趣召回，
门禁分支学习空间活动、晚归和作息状态，二者仅在高层通过行为融合门控合并。

热门偏向通过训练集LogQ校正、逆热门度损失、MMR和多兴趣配额缓解；输出自动包含
Coverage、Gini、Novelty、平均热门度及列表内外多样性，并生成
`fig/08_diversity_debias.png`。

零行为新用户可直接运行：

```powershell
python cold_start_infer.py --subject 计算机类 --gender 男 --grade 2019
```

训练阶段会随机遮蔽学习和线下行为模态，使画像塔具备缺失模态与冷启动兜底能力。

公式增强包括对称InfoNCE、VICReg、不确定性多任务加权和多尺度时间衰减混合核。
学习到的任务权重与时间核参数保存到
`result/learned_formula_parameters.json`、Excel的`Learned Formula Params`工作表，
并生成`fig/09_learned_formula_parameters.png`。
