"""Two-level Progressive Layered Extraction (PLE) for three ranking tasks."""
from __future__ import annotations

import torch
from torch import nn


TASKS = ("knowledge", "course", "consume")


def expert(input_dim, hidden, dropout):
    return nn.Sequential(
        nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
    )


class CGCLayer(nn.Module):
    """One customized-gate layer with task and shared information streams."""

    def __init__(self, input_dim, hidden, shared_experts, task_experts, dropout):
        super().__init__()
        self.shared_experts = nn.ModuleList(
            [expert(input_dim, hidden, dropout) for _ in range(shared_experts)])
        self.task_experts = nn.ModuleDict({
            task: nn.ModuleList([expert(input_dim, hidden, dropout) for _ in range(task_experts)])
            for task in TASKS
        })
        task_choices = task_experts + shared_experts
        shared_choices = len(TASKS) * task_experts + shared_experts
        self.task_gates = nn.ModuleDict({task: nn.Linear(input_dim, task_choices) for task in TASKS})
        self.shared_gate = nn.Linear(input_dim, shared_choices)

    @staticmethod
    def mix(values, logits):
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
        return (torch.stack(values, dim=1) * weights).sum(1)

    def forward(self, representations):
        shared_input = representations["shared"]
        shared_values = [module(shared_input) for module in self.shared_experts]
        task_values = {
            task: [module(representations[task]) for module in self.task_experts[task]]
            for task in TASKS
        }
        outputs = {}
        for task in TASKS:
            outputs[task] = self.mix(
                task_values[task] + shared_values,
                self.task_gates[task](representations[task]),
            )
        all_values = [value for task in TASKS for value in task_values[task]] + shared_values
        outputs["shared"] = self.mix(all_values, self.shared_gate(shared_input))
        return outputs


class PLEMultiTask(nn.Module):
    """Two progressive CGC levels followed by independent task towers."""

    TASKS = TASKS

    def __init__(self, input_dim: int, hidden: int, candidate_dim: int,
                 shared_experts: int, task_experts: int, dropout: float):
        super().__init__()
        self.levels = nn.ModuleList([
            CGCLayer(input_dim, hidden, shared_experts, task_experts, dropout),
            CGCLayer(hidden, hidden, shared_experts, task_experts, dropout),
        ])
        self.towers = nn.ModuleDict({
            task: nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
            for task in TASKS
        })
        self.rankers = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(hidden + candidate_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
            for task in TASKS
        })

    def forward(self, x, candidate_embeddings):
        task_inputs = {task: (x[task] if isinstance(x, dict) else x) for task in TASKS}
        shared_input = (x.get("shared") if isinstance(x, dict) and "shared" in x
                        else torch.stack(list(task_inputs.values()), dim=0).mean(0))
        representations = {**task_inputs, "shared": shared_input}
        for level in self.levels:
            representations = level(representations)
        result = {}
        for task in TASKS:
            task_hidden = self.towers[task](representations[task])
            candidates = candidate_embeddings[task]
            expanded = task_hidden.unsqueeze(1).expand(-1, candidates.size(1), -1)
            result[task] = self.rankers[task](torch.cat([expanded, candidates], dim=-1)).squeeze(-1)
        return result
