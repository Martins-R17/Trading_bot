"""Optional reinforcement-learning extension point.

The class is intentionally executable and dependency-light. It records
state/action/reward tuples and exposes the same shape a PyTorch agent could
later implement without changing the orchestration layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ReinforcementLearningAgent:
    """Small epsilon-greedy policy shell for future deep RL experiments."""

    actions: tuple[str, ...] = ("momentum", "mean_reversion", "breakout", "scalping_microstructure")
    epsilon: float = 0.05
    rewards: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for action in self.actions:
            self.rewards.setdefault(action, [])

    def choose_action(self, state: dict[str, Any]) -> str:
        if np.random.random() < self.epsilon:
            return str(np.random.choice(self.actions))
        averages = {action: np.mean(values) if values else 0.0 for action, values in self.rewards.items()}
        return max(averages, key=averages.get)

    def observe(self, action: str, reward: float, next_state: dict[str, Any] | None = None) -> None:
        if action in self.rewards:
            self.rewards[action].append(float(reward))

