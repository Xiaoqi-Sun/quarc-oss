"""
Minimal evaluation utilities for OSS pipeline.
Contains only the functions needed by the model evaluation and optimization scripts.
"""

import torch
from typing import Literal, Callable, Any
from torch import Tensor
import numpy as np
from collections import Counter
from dataclasses import dataclass

from src.quarc.models.modules.agent_encoder import AgentEncoder
from src.quarc.predictors.base import StagePrediction


@dataclass
class StageResult:
    """Results for individual stage"""
    is_correct: bool
    is_correct_relaxed: bool = False  # within-one-bin tolerance
    is_correct_2bins: bool = False  # within-two-bins tolerance


@dataclass
class OverallResult:
    """Overall results for a single reaction"""
    # Overall results
    is_fully_correct: bool
    is_fully_correct_relaxed: bool  # within-one-bin tolerance
    
    # Per-stage results
    agent_result: StageResult
    temperature_result: StageResult
    reactant_amount_result: StageResult
    agent_amount_result: StageResult


def is_correct_by_set(
    preds: Tensor | list[str] | list[int],
    targets: Tensor | list[str] | list[int],
    a_enc: AgentEncoder,
) -> bool:
    """
    1. use set match instead direct match for each predictions (not entire prediction as a whole set)
    2. if water is the only difference either way, count as correct
        e.g set(one_pred) - set(target) = 'O' or set(target) - set(one_pred) = 'O'
    """
    
    if isinstance(preds, Tensor):
        pred_indices = torch.nonzero(preds).squeeze(-1).tolist()
    else:
        pred_indices = preds

    if isinstance(targets, Tensor):
        target_indices = torch.nonzero(targets).squeeze(-1).tolist()
    else:
        target_indices = targets

    pred_names = a_enc.decode(pred_indices)
    target_names = a_enc.decode(target_indices)

    pred_set = set(pred_names)
    target_set = set(target_names)

    if pred_set == target_set:
        return True

    # Check if only water is different
    pred_diff = pred_set - target_set
    target_diff = target_set - pred_set

    if (pred_diff == {"O"} and not target_diff) or (target_diff == {"O"} and not pred_diff):
        return True

    return False


def is_correct_by_idx(
    preds: Tensor | list[str] | list[int],
    targets: Tensor | list[str] | list[int],
    a_enc: AgentEncoder,  # Not used but kept for consistency
) -> bool:
    """Direct index match for agent predictions"""
    if isinstance(preds, Tensor):
        pred_indices = torch.nonzero(preds).squeeze(-1).tolist()
    else:
        pred_indices = preds

    if isinstance(targets, Tensor):
        target_indices = torch.nonzero(targets).squeeze(-1).tolist()
    else:
        target_indices = targets

    return sorted(pred_indices) == sorted(target_indices)


def is_correct_by_combination(
    preds: Tensor | list[str] | list[int],
    targets: Tensor | list[str] | list[int],
    a_enc: AgentEncoder,
) -> bool:
    """Combination of set and idx matching - currently same as set"""
    return is_correct_by_set(preds, targets, a_enc)


def get_criteria_fn(
    criteria: Literal["set", "idx", "combination"],
) -> Callable[[Tensor, Tensor, AgentEncoder], bool]:
    """Get the criteria function (correct by set/idx/combination)."""
    if criteria == "set":
        return is_correct_by_set
    elif criteria == "idx":
        return is_correct_by_idx
    elif criteria == "combination":
        return is_correct_by_combination
    else:
        raise TypeError("criteria must be either 'set', 'idx', or 'combination'")


def check_agent_identity(predicted: list[str], target: list[str], agent_encoder) -> StageResult:
    """Evaluate agent identity prediction using relaxed set match"""
    is_correct_relaxed = is_correct_by_set(predicted, target, agent_encoder)
    is_correct_strict = sorted(predicted) == sorted(target)
    
    return StageResult(
        is_correct=is_correct_strict,
        is_correct_relaxed=is_correct_relaxed,
        is_correct_2bins=is_correct_relaxed  # Same as relaxed for agents
    )


def check_temperature(predicted: int, target: int) -> StageResult:
    """Evaluate temperature prediction using bin index"""
    is_correct = predicted == target
    is_within_one = abs(predicted - target) <= 1
    is_within_two = abs(predicted - target) <= 2

    return StageResult(
        is_correct=is_correct, 
        is_correct_relaxed=is_within_one, 
        is_correct_2bins=is_within_two
    )


def check_reactant_amounts(predicted: list[int], target: list[int]) -> StageResult:
    """Evaluate reactant amount prediction using reactant amount bin indices (unordered)"""
    is_correct = Counter(predicted) == Counter(target)

    # only model predictions guarantee the same length, baseline predictions may not
    if len(predicted) == len(target):
        is_within_one = all(np.abs(np.array(sorted(predicted)) - np.array(sorted(target))) <= 1)
        is_within_two = all(np.abs(np.array(sorted(predicted)) - np.array(sorted(target))) <= 2)
    else:
        is_within_one = False
        is_within_two = False

    return StageResult(
        is_correct=is_correct, 
        is_correct_relaxed=is_within_one, 
        is_correct_2bins=is_within_two
    )


def check_agent_amounts(
    predicted: list[tuple[int, int]], target: list[tuple[int, int]]
) -> StageResult:
    """Evaluate agent amount prediction using mapped agent indices and bin indices"""
    # Sort both by agent index for comparison
    pred_sorted = sorted(predicted, key=lambda x: x[0])
    target_sorted = sorted(target, key=lambda x: x[0])

    is_correct = pred_sorted == target_sorted

    # Check within-one-bin tolerance for amounts (second element of tuple)
    if len(pred_sorted) == len(target_sorted):
        is_within_one = all(
            abs(p[1] - t[1]) <= 1 for p, t in zip(pred_sorted, target_sorted)
            if p[0] == t[0]  # Same agent
        )
        is_within_two = all(
            abs(p[1] - t[1]) <= 2 for p, t in zip(pred_sorted, target_sorted)
            if p[0] == t[0]  # Same agent
        )
    else:
        is_within_one = False
        is_within_two = False

    return StageResult(
        is_correct=is_correct, 
        is_correct_relaxed=is_within_one, 
        is_correct_2bins=is_within_two
    )


def check_overall_prediction(
    stage_pred: StagePrediction,
    targets: dict[str, Any],
    agent_encoder,
) -> OverallResult:
    """Check all stages of a prediction against targets."""
    # Stage 1: Agents
    agent_result = check_agent_identity(stage_pred.agents, targets["target_agents"], agent_encoder)
    # Stage 2: Temperature
    temp_result = check_temperature(stage_pred.temp_bin, targets["target_temp"])
    # Stage 3: Reactant amounts
    reactant_result = check_reactant_amounts(
        stage_pred.reactant_bins, targets["target_reactant_amounts"]
    )
    # Stage 4: Agent amounts
    agent_amount_result = check_agent_amounts(
        stage_pred.agent_amount_bins, targets["target_agent_amounts"]
    )
    
    # Standard stage 1 evaluation uses relaxed check (is_correct_by_set)
    is_fully_correct = (
        agent_result.is_correct_relaxed
        and temp_result.is_correct
        and reactant_result.is_correct
        and agent_amount_result.is_correct
    )
    
    is_fully_correct_relaxed = (
        agent_result.is_correct_relaxed
        and temp_result.is_correct_relaxed
        and reactant_result.is_correct_relaxed
        and agent_amount_result.is_correct_relaxed
    )

    return OverallResult(
        is_fully_correct=is_fully_correct,
        is_fully_correct_relaxed=is_fully_correct_relaxed,
        agent_result=agent_result,
        temperature_result=temp_result,
        reactant_amount_result=reactant_result,
        agent_amount_result=agent_amount_result,
    )