from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(unsafe_hash=True)
class Hyperparameters:
    policy_learning_rate : float = 1e-8
    value_learning_rate : float = 3e-8
    policy_trace_decay : float = 0.85
    value_trace_decay : float = 0.85
    discount_factor : float = 0.92
    max_td_error_mag : float = 2.0
    max_value_trace : float = 8.0
    max_policy_trace : float = 8.0
    max_value_weight_update : float = 0.01
    max_policy_weight_update : float = 0.01
    max_policy_grad_norm :  float = 2.0
    max_value_grad_norm : float = 1.0
    discount_factor_decay : Optional[float] = 1
    value_function_changeout : Optional[int] = 500
    max_episodes : int = 1000
