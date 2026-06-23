from __future__ import annotations

from pathlib import Path


def write_thesis_text(path: Path, config, artifacts, test_metrics, ablation):
    mean_ndcg = sum(test_metrics.get(f"{t}_final_ndcg@10", test_metrics[f"{t}_ndcg@10"])
                    for t in ("knowledge", "course", "consume")) / 3
    full = ablation["完整模型"]
    baselines = [v for k, v in ablation.items() if k != "完整模型"]
    baseline_best = max((sum(x[f"{t}_ndcg@10"] for t in ("knowledge", "course", "consume"))/3 for x in baselines), default=0)
    has_baseline = bool(baselines)
    improvement = (mean_ndcg - baseline_best) / max(baseline_best, 1e-8) * 100 if has_baseline else None
    comparison_text = (f"相较最优消融基线，平均相对变化为{improvement:.2f}%。"
                       if has_baseline else "本次为跳过消融的调试运行，不报告消融提升比例。")
    text = f"""# 可直接用于论文的实验文字

## 模型架构描述
本文构建TransE–时间感知AKT–双分支兴趣漂移编码–个体跨域映射–双塔召回–PLE精排联合推荐框架。题目知识点采用直接标注、课程链路、EXERCISE_UNIT与PROBLEM_UNIT的分级映射。AKT融合Time2Vec、相对位置编码与时间衰减因果注意力。课程任务使用目标课程选课时刻之前的独立答题切片，防止选课后学习行为进入课程预测。消费与门禁分别由独立DIEN和AUGRU编码，再融合为真实行为表征z_real；完整模型通过可学习门控将z_real与预测行为表征z_behavior融合并直接送入消费推荐塔，因此门禁状态能够影响最终推荐。在行为缺失时，模型自动退回由z_stu映射得到的z_behavior。双塔召回Top-{config.retrieval_topk}候选，PLE随后仅对候选执行多任务精排。
课程精排优先使用数据集确定性分级映射：依次建立Concept→Course、Exercise→Course及Problem→Exercise→Course关系，显式关系缺失时才回退TransE相似扩展。AKT预测的知识掌握概率用于计算薄弱覆盖，历史课程用于兴趣连续性，正确知识表征用于图谱先修准备度，课程知识规模用于难度适配。显式映射、TransE薄弱扩展、兴趣、先修和难度权重分别为{config.course_explicit_mapping_weight}、{config.course_weakness_weight}、{config.course_interest_weight}、{config.course_prerequisite_weight}和{config.course_difficulty_weight}，组合分数以{config.course_logic_scale}倍加入课程精排结果；知识单元与课程仍分别输出独立Top-K列表，已选课程在最终阶段过滤。

## 实验设置
融合数据包含{len({x.user_id for x in artifacts.samples})}名统一ID用户，共构造{len(artifacts.samples)}条事件级样本。消费任务采用滑动窗口预测下一消费品类，每名用户最多保留{config.max_consume_windows_per_user}个近期窗口；课程任务将每次后续选课分别作为目标，并严格使用该次选课前的答题与课程历史。实验按原始泰迪卡号分组并按学科划分为70%训练集、15%验证集和15%测试集，同一原始卡及其全部窗口不会跨集合出现。嵌入维度为{config.embedding_dim}，隐藏维度为{config.hidden_dim}，批大小为{config.batch_size}，初始学习率为{config.learning_rate}，Dropout为{config.dropout}。采用AdamW优化器与ReduceLROnPlateau学习率调度，验证集平均NDCG@10连续{config.early_stopping_patience}轮未提升时早停。评价指标包括双塔Retrieval Recall@{config.retrieval_topk}以及最终排序的AUC、NDCG@5、NDCG@10、Recall@5和Recall@10。召回失败样本由召回损失负责，PLE精排损失仅在真实目标进入候选集时计算。

## 实验结果分析
完整模型在测试集三任务上的平均NDCG@10为{mean_ndcg:.4f}。{comparison_text}消融结果用于检验个体级行为对齐与跨域映射的贡献。各任务的详细结果见metric_result.xlsx。

为缓解热门偏向，双塔分数采用训练集LogQ热门校正，推荐损失采用逆热门度加权；PLE精排后进一步使用MMR和多兴趣配额进行列表重排。实验同时报告Coverage、Gini、Novelty、Average Popularity、列表内多样性和列表间多样性。冷启动方面，训练阶段随机遮蔽学习、消费和门禁模态，使画像塔学习零行为兜底；零行为推理将画像模型结果与训练集热门先验按{config.cold_start_popularity_mix:.2f}比例混合，再执行去偏与MMR重排。

数学目标方面，跨域个体对齐采用双向InfoNCE，使同一global_user_id的学业预测向量与真实行为向量互为正样本、批内其他用户为负样本；VICReg进一步通过不变性、方差下界和去协方差项抑制表征坍缩。PLE三任务采用可学习同方差不确定性加权，即对任务t优化0.5exp(-s_t)L_t+0.5s_t。时间注意力采用{config.time_decay_kernels}个指数核的凸组合K(Δt)=Σ_k α_k exp(-λ_kΔt)，分别学习短时、中时和长期记忆尺度。

## 图注
图1 训练集与验证集联合损失随训练轮次的变化。\n
图2 三项推荐任务在验证集上的NDCG@5、NDCG@10与AUC变化。\n
图3 完整模型与两种跨域映射消融模型的测试集平均NDCG@10对比。\n
图4 八大学科有效样本数量及占比。\n
图5 学业表征z_stu与映射行为表征z_behavior的t-SNE可视化。\n
图6 随机测试用户的知识点、课程与消费品类Top-K推荐结果。

图7 测试用户消费兴趣、门禁状态漂移分布及消费短期兴趣门控关系。

图8 三项任务的推荐覆盖率、Gini系数及列表内外多样性。

图9 模型学习得到的任务权重与多尺度时间衰减核参数。

## 研究局限性
模型在实验计算中将共享global_user_id的MOOC记录与一卡通记录直接视为同一学生，并进行个体级表征对齐；但该global_user_id实际由两套独立数据依据学科分层规则合成，并非可验证的真实身份关联。因此，本实验只能说明所提方法在“合成同一学生”假设下的可行性，不能据此推断现实个体的跨域行为关系或因果关系。MOOCCubeX用户学科由选课course_order反向统计课程field得到，并非用户原生专业字段。此外，课程链路回退与EXERCISE_UNIT属于自动构造的弱知识标签，其精度低于concept-problem直接标注。后续研究应在获得授权、完成匿名化和伦理审查的真实同源校园数据上复验。
"""
    path.write_text(text, encoding="utf-8")
    print(text)
