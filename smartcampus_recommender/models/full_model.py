from __future__ import annotations

import torch
from torch import nn

from config import ExperimentConfig
from data.preprocessing import DataArtifacts
from models.akt import AKTEncoder
from models.cross_domain import PersonalizedMapper
from models.dien import DIENEncoder
from models.ple import PLEMultiTask
from models.transe import TransEEncoder
from models.two_tower import TwoTowerRetriever
from models.interest_drift import InterestDriftModule


class SmartCampusRecommender(nn.Module):
    """TransE → AKT/DIEN → 跨域个性化映射 → PLE 三任务推荐。"""
    def __init__(self, artifacts: DataArtifacts, config: ExperimentConfig):
        super().__init__()
        v, d = artifacts.vocabs, config.embedding_dim
        self.retrieval_topk = config.retrieval_topk
        self.course_rule_candidates = config.course_rule_candidates
        self.popularity_penalty = config.popularity_penalty
        self.mmr_diversity_weight = config.mmr_diversity_weight
        self.rerank_pool_size = config.rerank_pool_size
        self.interest_quota_per_vector = config.interest_quota_per_vector
        self.course_logic_scale = config.course_logic_scale
        self.course_score_temperature_floor = config.course_score_temperature_floor
        self.course_fusion_logits = nn.Parameter(torch.log(torch.tensor([0.45, 0.35, 0.20])))
        self.course_log_temperatures = nn.Parameter(torch.zeros(3))
        self.course_logic_weights = {
            "explicit": config.course_explicit_mapping_weight,
            "weakness": config.course_weakness_weight,
            "interest": config.course_interest_weight,
            "prerequisite": config.course_prerequisite_weight,
            "difficulty": config.course_difficulty_weight,
        }
        self.subject_emb = nn.Embedding(len(v["subject"]), d, padding_idx=0)
        self.gender_emb = nn.Embedding(len(v["gender"]), d, padding_idx=0)
        self.grade_emb = nn.Embedding(len(v["grade"]), d, padding_idx=0)
        self.window_hour_emb = nn.Embedding(24, d)
        self.window_weekday_emb = nn.Embedding(7, d)
        self.window_holiday_emb = nn.Embedding(2, d)
        self.window_context_norm = nn.LayerNorm(d)
        self.static_context = nn.Parameter(torch.zeros(d))
        self.profile = nn.Sequential(nn.Linear(d * 3, d), nn.GELU(), nn.LayerNorm(d))
        self.transe = TransEEncoder(len(v["kg_entity"]), len(v["kg_relation"]), d, config.transe_margin)
        self.has_concept_relation_idx = v["kg_relation"].encode("has_concept")
        self.register_buffer("concept_to_kg", torch.tensor(
            [v["kg_entity"].encode(token) for token in v["concept"].idx_to_token], dtype=torch.long))
        self.register_buffer("course_to_kg", torch.tensor(
            [v["kg_entity"].encode(token) for token in v["course"].idx_to_token], dtype=torch.long))
        self.register_buffer("popularity_knowledge", torch.zeros(len(v["concept"])))
        self.register_buffer("popularity_course", torch.zeros(len(v["course"])))
        self.register_buffer("popularity_consume", torch.zeros(len(v["consume"])))
        self.register_buffer("course_difficulty", torch.tensor(artifacts.course_difficulty, dtype=torch.float32))
        if artifacts.knowledge_course_edges:
            edge_tensor = torch.tensor(artifacts.knowledge_course_edges, dtype=torch.float32)
            hierarchy_indices = torch.stack([edge_tensor[:, 1].long(), edge_tensor[:, 0].long()])
            hierarchy_values = edge_tensor[:, 2]
        else:
            hierarchy_indices = torch.empty((2, 0), dtype=torch.long)
            hierarchy_values = torch.empty((0,), dtype=torch.float32)
        self.register_buffer("hierarchy_indices", hierarchy_indices)
        self.register_buffer("hierarchy_values", hierarchy_values)
        self.akt = AKTEncoder(len(v["question"]), len(v["concept"]), d,
                             config.num_heads, config.akt_layers, config.dropout,
                             config.max_learning_len, config.time_decay_kernels)
        self.consume_dien = DIENEncoder(
            len(v["consume"]), d, config.dropout, use_internal_item_embedding=False)
        self.door_dien = DIENEncoder(len(v["door_behavior"]), d, config.dropout)
        self.consume_candidate_fusion = nn.Sequential(
            nn.Linear(d * 2, d), nn.GELU(), nn.LayerNorm(d))
        self.consume_interest_drift = InterestDriftModule(
            d, config.num_interests, config.short_interest_window, config.dropout,
            config.drift_recent_hours, config.drift_history_hours)
        self.door_interest_drift = InterestDriftModule(
            d, max(2, config.num_interests // 2), config.short_interest_window, config.dropout,
            config.drift_recent_hours, config.drift_history_hours)
        self.behavior_fusion_gate = nn.Sequential(
            nn.Linear(d * 3 + 2, d), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(d, 1), nn.Sigmoid())
        self.behavior_fusion_norm = nn.LayerNorm(d)
        self.student_fusion = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.LayerNorm(d))
        self.mapper = PersonalizedMapper(d, config.hidden_dim, config.dropout, input_dim=d*2)
        self.real_behavior_gate = nn.Sequential(
            nn.Linear(d * 3, d), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(d, 1), nn.Sigmoid())
        # DIEN自监督辅助头仅用于让真实线下表征学习消费兴趣；跨域映射仍只读取学科均值。
        self.behavior_aux_bias = nn.Parameter(torch.zeros(len(v["consume"])))
        # 全局均值仅用于消融对照；完整模型执行同一global_user_id的个体映射。
        self.register_buffer("global_behavior_base", torch.zeros(d))
        self.retriever = TwoTowerRetriever(
            user_input_dim=d * 2, dim=d,
            candidate_sizes={"knowledge": len(v["concept"]), "course": len(v["course"]),
                             "consume": len(v["consume"])},
            temperature=config.retrieval_temperature, dropout=config.dropout)
        self.ple = PLEMultiTask(
            input_dim=d * 2, hidden=config.hidden_dim, candidate_dim=d,
            shared_experts=config.num_shared_experts, task_experts=config.num_task_experts,
            dropout=config.dropout,
        )
        # s_t=log σ_t²，由数据自动学习知识点、课程、消费任务的噪声/难度。
        task_weights = torch.tensor([
            config.task_knowledge_weight, config.task_course_weight, config.task_consume_weight
        ], dtype=torch.float32)
        self.register_buffer("task_weights", task_weights / task_weights.sum().clamp_min(1e-8))

    def profile_vector(self, batch):
        return self.profile(torch.cat([
            self.subject_emb(batch["subject"]), self.gender_emb(batch["gender"]),
            self.grade_emb(batch["grade"]),
        ], dim=-1))

    def encode_behavior(self, batch):
        profile = self.profile_vector(batch)
        behavior = self.encode_split_behavior(batch, profile)
        return behavior["z_real"]

    def encode_split_behavior(self, batch, profile):
        consume_zero = torch.zeros_like(batch["consume_weekly"])
        consume_long, consume_attention, consume_evolved = self.consume_dien(
            batch["consume_behavior"], batch["consume_meal"], consume_zero,
            batch["consume_weekly"], batch["consume_hour"], batch["consume_weekday"],
            batch["consume_holiday"], batch["consume_mask"], profile,
            shared_item_weight=self.retriever.candidates["consume"].weight)
        consume_drift = self.consume_interest_drift(
            consume_long, consume_evolved, batch["consume_mask"], profile, batch["consume_weekly"],
            batch["consume_age_hours"])

        door_meal = torch.zeros_like(batch["door_behavior"])
        door_long, door_attention, door_evolved = self.door_dien(
            batch["door_behavior"], door_meal, batch["door_late"], batch["door_weekly"],
            batch["door_hour"], batch["door_weekday"], batch["door_holiday"],
            batch["door_mask"], profile)
        door_drift = self.door_interest_drift(
            door_long, door_evolved, batch["door_mask"], profile, batch["door_weekly"],
            batch["door_age_hours"])

        consume_valid = batch["consume_mask"].any(1).float().unsqueeze(-1)
        door_valid = batch["door_mask"].any(1).float().unsqueeze(-1)
        fusion_gate = self.behavior_fusion_gate(torch.cat([
            consume_drift["dynamic_interest"], door_drift["dynamic_interest"], profile,
            consume_valid, door_valid], dim=-1))
        # 缺失某一分支时强制使用可用分支；两者均有时由门控学习。
        fusion_gate = torch.where((consume_valid > 0) & (door_valid == 0), torch.ones_like(fusion_gate), fusion_gate)
        fusion_gate = torch.where((consume_valid == 0) & (door_valid > 0), torch.zeros_like(fusion_gate), fusion_gate)
        z_real = self.behavior_fusion_norm(
            fusion_gate * consume_drift["dynamic_interest"] +
            (1.0 - fusion_gate) * door_drift["dynamic_interest"])
        return {"z_real": z_real, "consume": consume_drift, "door": door_drift,
                "consume_long": consume_long, "door_long": door_long,
                "consume_sequence": consume_evolved,
                "consume_attention": consume_attention, "door_attention": door_attention,
                "behavior_fusion_gate": fusion_gate.squeeze(-1)}

    def forward(self, batch, mode: str = "full"):
        profile = self.profile_vector(batch)
        z_seq, kt_logits = self.akt(
            batch["questions"], batch["concepts"], batch["correct"], batch["intervals"],
            batch["night"], batch["wrong_streak"], batch["relative_position"], batch["learning_mask"],
            self.transe.entity(self.concept_to_kg[batch["concepts"]]))
        z_stu = self.student_fusion(torch.cat([z_seq, profile], dim=-1))
        # 课程塔使用目标课程选课前的专属学习切片，杜绝选课后的答题信息泄漏。
        z_course_seq, course_kt_logits = self.akt(
            batch["course_questions"], batch["course_concepts"], batch["course_correct"],
            batch["course_intervals"], batch["course_night"], batch["course_wrong_streak"],
            batch["course_relative_position"], batch["course_learning_mask"],
            self.transe.entity(self.concept_to_kg[batch["course_concepts"]]))
        z_stu_course = self.student_fusion(torch.cat([z_course_seq, profile], dim=-1))
        behavior_output = self.encode_split_behavior(batch, profile)
        consume_drift = behavior_output["consume"]
        door_drift = behavior_output["door"]
        z_real = behavior_output["z_real"]
        window_context = self.window_context_norm(
            self.window_hour_emb(batch["context_hour"].clamp(0, 23)) +
            self.window_weekday_emb(batch["context_weekday"].clamp(0, 6)) +
            self.window_holiday_emb(batch["context_holiday"].clamp(0, 1)))
        # A learnable no-timestamp token is optimized by the knowledge/course
        # objectives. A literal zero vector is an out-of-distribution context
        # for a mapper otherwise trained with normalized time embeddings.
        static_context = self.static_context.unsqueeze(0).expand_as(window_context)
        if mode == "global_mean":
            z_behavior = self.global_behavior_base.unsqueeze(0).expand_as(z_stu)
        elif mode == "no_cross":
            z_behavior = torch.zeros_like(z_stu)
        else:
            z_behavior = self.mapper(torch.cat([z_stu, window_context], dim=-1))
        z_behavior_static = (self.mapper(torch.cat([z_stu, static_context], dim=-1))
                             if mode == "full" else z_behavior)
        z_behavior_course = (self.mapper(torch.cat([z_stu_course, static_context], dim=-1))
                             if mode == "full" else z_behavior)
        behavior_available = (batch["consume_mask"].any(1) | batch["door_mask"].any(1)).unsqueeze(-1)
        real_gate = self.real_behavior_gate(torch.cat([z_stu, z_behavior, z_real], dim=-1))
        # 真实消费+门禁表征直接进入消费推荐；无行为或消融模式自动退回预测行为表征。
        z_behavior_online = (torch.where(
            behavior_available, real_gate*z_real + (1.0-real_gate)*z_behavior, z_behavior)
            if mode == "full" else z_behavior)
        user_fused = {
            "knowledge": torch.cat([z_stu, z_behavior_static], dim=-1),
            "course": torch.cat([z_stu_course, z_behavior_course], dim=-1),
            "consume": torch.cat([z_stu, z_behavior_online], dim=-1),
            # Profile is the only observation shared by all three task clocks.
            # Task-specific temporal states never cross into another task here.
            "shared": torch.cat([profile, profile], dim=-1),
        }
        # TransE图谱向量与可学习候选向量共同构成物品塔，双塔先召回Top-N。
        knowledge_entities = self.transe.entity(self.concept_to_kg)
        course_entities = self.transe.entity(self.course_to_kg)
        course_matrix = self.retriever.candidate_matrix("course", course_entities)
        normalized_courses = torch.nn.functional.normalize(course_matrix, dim=-1)
        # In TransE, course + has_concept is comparable with a concept; raw
        # course/concept cosine similarity ignores the learned relation.
        has_concept_relation = self.transe.relation.weight[self.has_concept_relation_idx]
        normalized_course_graph = torch.nn.functional.normalize(
            course_entities + has_concept_relation.unsqueeze(0), dim=-1)
        history_concepts = self.transe.entity(self.concept_to_kg[batch["course_concepts"]])
        history_mask = batch["course_learning_mask"].float()

        # AKT在位置t预测t+1正确率，将其对齐为后续知识单元的掌握概率。
        mastery_probability = batch["course_correct"].clone()
        if mastery_probability.size(1) > 1:
            mastery_probability[:, 1:] = torch.sigmoid(course_kt_logits[:, :-1])

        def weighted_history_vector(weight):
            weight = weight * history_mask
            vector = (history_concepts * weight.unsqueeze(-1)).sum(1) / weight.sum(1, keepdim=True).clamp_min(1.0)
            return torch.nn.functional.normalize(vector, dim=-1), weight.sum(1).gt(0)

        weak_vector, weak_valid = weighted_history_vector(1.0-mastery_probability)
        mastered_vector, mastery_valid = weighted_history_vector(mastery_probability)
        weak_match = (weak_vector @ normalized_course_graph.T + 1.0) / 2.0
        prerequisite = (mastered_vector @ normalized_course_graph.T + 1.0) / 2.0
        weak_match = torch.where(weak_valid.unsqueeze(1), weak_match, torch.full_like(weak_match, 0.5))
        prerequisite = torch.where(mastery_valid.unsqueeze(1), prerequisite,
                                   torch.full_like(prerequisite, 0.5))

        # 固定分级映射优先：Concept/Exercise/Problem映射边直接聚合AKT薄弱度。
        batch_size = batch["course_concepts"].size(0)
        knowledge_size = self.concept_to_kg.numel()
        weak_sum = history_mask.new_zeros((batch_size, knowledge_size))
        occurrence = history_mask.new_zeros((batch_size, knowledge_size))
        weak_sum.scatter_add_(1, batch["course_concepts"],
                              (1.0-mastery_probability)*history_mask)
        occurrence.scatter_add_(1, batch["course_concepts"], history_mask)
        weak_by_knowledge = weak_sum / occurrence.clamp_min(1.0)
        observed = occurrence.gt(0).float()
        # CUDA sparse addmm has no FP16 kernel. Keep only the sparse hierarchy
        # propagation in FP32 while the dense encoders/rankers remain under AMP.
        with torch.autocast(device_type=course_matrix.device.type, enabled=False):
            hierarchy = torch.sparse_coo_tensor(
                self.hierarchy_indices, self.hierarchy_values.float(),
                size=(course_matrix.size(0), knowledge_size),
                device=course_matrix.device, check_invariants=False).coalesce()
            weak_fp32 = weak_by_knowledge.float()
            observed_fp32 = observed.float()
            explicit_numerator = torch.sparse.mm(hierarchy, weak_fp32.T).T
            explicit_denominator = torch.sparse.mm(hierarchy, observed_fp32.T).T
            explicit_available = explicit_denominator.gt(0)
            explicit_coverage = explicit_numerator / explicit_denominator.clamp_min(1e-8)
            explicit_mastery_numerator = torch.sparse.mm(
                hierarchy, ((1.0 - weak_fp32)*observed_fp32).T).T
            explicit_mastery = explicit_mastery_numerator / explicit_denominator.clamp_min(1e-8)
        prerequisite = torch.where(explicit_available, explicit_mastery, prerequisite)

        history_course_vectors = course_matrix[batch["course_history"]]
        course_history_mask = batch["course_history_mask"].float()
        interest_vector = (history_course_vectors * course_history_mask.unsqueeze(-1)).sum(1) / \
                          course_history_mask.sum(1, keepdim=True).clamp_min(1.0)
        interest_vector = torch.nn.functional.normalize(interest_vector, dim=-1)
        interest = (interest_vector @ normalized_courses.T + 1.0) / 2.0
        interest = torch.where(course_history_mask.any(1).unsqueeze(1), interest,
                               torch.full_like(interest, 0.5))
        ability = (mastery_probability*history_mask).sum(1) / history_mask.sum(1).clamp_min(1.0)
        ability = torch.where(history_mask.any(1), ability, torch.full_like(ability, 0.5))
        difficulty_fit = torch.exp(-3.0*torch.abs(ability.unsqueeze(1)-self.course_difficulty.unsqueeze(0)))
        hierarchy_component = torch.where(
            explicit_available,
            self.course_logic_weights["explicit"]*explicit_coverage +
            self.course_logic_weights["weakness"]*weak_match,
            (self.course_logic_weights["explicit"]+self.course_logic_weights["weakness"])*weak_match)
        course_logic_score = (
            hierarchy_component +
            self.course_logic_weights["interest"]*interest +
            self.course_logic_weights["prerequisite"]*prerequisite +
            self.course_logic_weights["difficulty"]*difficulty_fit)
        targets = {"knowledge": batch["target_concept"], "course": batch["target_course"],
                   "consume": batch["target_consume"]}
        popularity = {
            "knowledge": self.popularity_knowledge,
            "course": self.popularity_course,
            "consume": self.popularity_consume,
        }
        retrieval_logits, candidate_indices, candidate_embeddings = self.retriever(
            user_fused,
            {"knowledge": knowledge_entities, "course": course_entities, "consume": None},
            topk=self.retrieval_topk,
            targets=targets, force_target=self.training,
            multi_interests=consume_drift["multi_interests"],
            multi_interest_mask=consume_drift["multi_interest_mask"],
            exclusion_indices={"course": batch["course_history"]},
            exclusion_masks={"course": batch["course_history_mask"]})
        semantic_candidate_indices = dict(candidate_indices)
        # Rules are an independent source of candidates and cannot change the
        # semantic two-tower logits or its retrieval objective.
        rule_count = min(self.course_rule_candidates, max(course_logic_score.size(1) - 2, 0))
        if rule_count > 0:
            rule_scores = course_logic_score.clone()
            rule_scores[:, :2] = -1e4
            blocked_courses = torch.zeros_like(rule_scores, dtype=torch.bool)
            blocked_courses.scatter_(1, batch["course_history"].clamp(0, rule_scores.size(1)-1),
                                     batch["course_history_mask"] & batch["course_history"].gt(1))
            rule_scores = rule_scores.masked_fill(blocked_courses, -1e4)
            rule_indices = torch.topk(rule_scores, k=rule_count, dim=1).indices
            candidate_indices["course"] = torch.cat([candidate_indices["course"], rule_indices], dim=1)
            candidate_embeddings["course"] = course_matrix[candidate_indices["course"]]
        target_interest, target_attention = self.consume_dien.evolve_for_candidates(
            behavior_output["consume_sequence"], batch["consume_mask"],
            candidate_embeddings["consume"])
        candidate_embeddings["consume"] = self.consume_candidate_fusion(torch.cat([
            candidate_embeddings["consume"], target_interest], dim=-1))
        consume_ranking_matrix = candidate_embeddings["consume"].new_zeros(
            candidate_embeddings["consume"].size(0), retrieval_logits["consume"].size(1),
            candidate_embeddings["consume"].size(-1))
        consume_ranking_matrix.scatter_(
            1, candidate_indices["consume"].unsqueeze(-1).expand_as(candidate_embeddings["consume"]),
            candidate_embeddings["consume"])
        # LogQ式热门校正：高曝光候选在召回阶段受到可控惩罚。
        rerank_scores = self.ple(user_fused, candidate_embeddings)
        # PLE只精排召回候选；候选外分数保持极低。召回分数作为精排先验共同参与排序。
        final_logits = {}
        for task in rerank_scores:
            retrieval_score = retrieval_logits[task].gather(1, candidate_indices[task])
            candidate_score = rerank_scores[task] + retrieval_score
            if task == "course":
                logic_score = course_logic_score.gather(1, candidate_indices[task])
                components = torch.stack([retrieval_score, rerank_scores[task], logic_score], dim=-1)
                mean = components.mean(1, keepdim=True)
                std = components.std(1, keepdim=True, unbiased=False).clamp_min(1e-4)
                normalized_components = (components - mean) / std
                temperatures = (torch.nn.functional.softplus(self.course_log_temperatures) +
                                self.course_score_temperature_floor)
                fusion_weights = torch.softmax(self.course_fusion_logits, dim=0)
                candidate_score = (normalized_components / temperatures * fusion_weights).sum(-1)
            full = torch.full(
                retrieval_logits[task].shape, -1e4, dtype=candidate_score.dtype,
                device=candidate_score.device)
            full.scatter_(1, candidate_indices[task], candidate_score)
            final_logits[task] = full
        return {**final_logits, "retrieval_logits": retrieval_logits, "popularity": popularity,
                "task_weights": self.task_weights,
                "candidate_indices": candidate_indices,
                "semantic_candidate_indices": semantic_candidate_indices,
                "kt_logits": kt_logits, "z_stu": z_stu,
                "z_behavior": z_behavior, "z_real": z_real,
                "z_behavior_static": z_behavior_static,
                "z_stu_course": z_stu_course, "z_behavior_online": z_behavior_online,
                "real_behavior_gate": real_gate.squeeze(-1),
                "course_logic_score": course_logic_score,
                "course_weakness_score": weak_match,
                "course_explicit_coverage_score": explicit_coverage,
                "course_explicit_mapping_available": explicit_available,
                "course_interest_score": interest,
                "course_prerequisite_score": prerequisite,
                "course_difficulty_score": difficulty_fit,
                "course_fusion_weights": torch.softmax(self.course_fusion_logits, dim=0),
                "course_fusion_temperatures": (torch.nn.functional.softplus(
                    self.course_log_temperatures) + self.course_score_temperature_floor),
                "z_long": behavior_output["consume_long"],
                "short_interest": consume_drift["short_interest"],
                "multi_interests": consume_drift["multi_interests"],
                "multi_interest_weights": consume_drift["multi_interest_weights"],
                "multi_interest_mask": consume_drift["multi_interest_mask"],
                "drift_score": consume_drift["drift_score"],
                "short_gate": consume_drift["short_gate"],
                "consume_drift_score": consume_drift["drift_score"],
                "door_drift_score": door_drift["drift_score"],
                "behavior_fusion_gate": behavior_output["behavior_fusion_gate"],
                "behavior_aux_logits": torch.nn.functional.linear(
                    z_real, self.retriever.candidates["consume"].weight, self.behavior_aux_bias),
                "behavior_attention": behavior_output["consume_attention"],
                "consume_target_attention": target_attention,
                "ranking_item_embeddings": {"consume": consume_ranking_matrix}}
