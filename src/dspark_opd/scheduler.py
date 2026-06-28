from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ScheduleStep:
    request_id: int
    prefix_len: int
    survival_probability: float
    batch_size: int
    expected_accepts: float
    throughput: float
    accepted: bool


@dataclass(frozen=True)
class PrefixSchedule:
    lengths: np.ndarray
    best_throughput: float
    expected_accepts: float
    batch_size: int
    trace: tuple[ScheduleStep, ...] = field(default_factory=tuple)


def _lookup_sps(sps_curve: dict[int, float] | list[float] | np.ndarray, batch_size: int) -> float:
    if isinstance(sps_curve, dict):
        if batch_size in sps_curve:
            return float(sps_curve[batch_size])
        candidates = [key for key in sps_curve if key <= batch_size]
        if candidates:
            return float(sps_curve[max(candidates)])
        return float(sps_curve[min(sps_curve)])
    curve = np.asarray(sps_curve, dtype=np.float64)
    if curve.ndim != 1:
        raise ValueError("sps_curve must be one-dimensional.")
    if len(curve) == 0:
        raise ValueError("sps_curve must not be empty.")
    idx = min(max(int(batch_size), 0), len(curve) - 1)
    return float(curve[idx])


def survival_probabilities(confidence: np.ndarray) -> np.ndarray:
    """Convert conditional confidence c_k to prefix survival a_k."""

    confidence = np.asarray(confidence, dtype=np.float64)
    if confidence.ndim != 2:
        raise ValueError("confidence must have shape [requests, block_size].")
    confidence = np.clip(confidence, 0.0, 1.0)
    return np.cumprod(confidence, axis=1)


def expected_throughput(
    lengths: np.ndarray,
    survival: np.ndarray,
    sps_curve: dict[int, float] | list[float] | np.ndarray,
    *,
    include_bonus_token: bool = True,
) -> tuple[float, float, int]:
    """Return throughput, expected accepts, and target verification batch size."""

    lengths = np.asarray(lengths, dtype=np.int64)
    survival = np.asarray(survival, dtype=np.float64)
    if survival.ndim != 2:
        raise ValueError("survival must have shape [requests, block_size].")
    if lengths.shape != (survival.shape[0],):
        raise ValueError("lengths must have shape [requests].")
    base = float(survival.shape[0]) if include_bonus_token else 0.0
    expected_accepts = base
    for request_id, length in enumerate(lengths):
        if length > 0:
            expected_accepts += float(np.sum(survival[request_id, : int(length)]))
    batch_size = int(np.sum(lengths) + (survival.shape[0] if include_bonus_token else 0))
    throughput = expected_accepts * _lookup_sps(sps_curve, batch_size)
    return float(throughput), float(expected_accepts), batch_size


def hardware_aware_prefix_scheduler(
    confidence: np.ndarray,
    sps_curve: dict[int, float] | list[float] | np.ndarray,
    *,
    early_stop: bool = True,
    min_survival_probability: float = 0.0,
    include_bonus_token: bool = True,
) -> PrefixSchedule:
    """DSpark-style hardware-aware prefix scheduler.

    Args:
        confidence: Conditional acceptance estimates with shape
            ``[num_requests, block_size]``.
        sps_curve: Steps-per-second table indexed by target verification batch
            size. A dict may be sparse; the nearest lower key is used.
        early_stop: Preserve the paper's non-anticipating scheduler. Turning this
            off is useful only when the capacity limit comes from an older causal
            signal, as described in the production adaptation.
    """

    survival = survival_probabilities(confidence)
    num_requests, block_size = survival.shape
    lengths = np.zeros(num_requests, dtype=np.int64)
    base_batch = num_requests if include_bonus_token else 0
    expected_accepts = float(num_requests) if include_bonus_token else 0.0
    best_throughput = expected_accepts * _lookup_sps(sps_curve, base_batch)
    best_expected_accepts = expected_accepts
    best_batch_size = base_batch
    best_lengths = lengths.copy()
    trace: list[ScheduleStep] = []

    candidates: list[tuple[float, int, int]] = []
    for request_id in range(num_requests):
        for pos in range(block_size):
            prob = float(survival[request_id, pos])
            if prob > min_survival_probability:
                candidates.append((prob, request_id, pos + 1))
    candidates.sort(key=lambda item: item[0], reverse=True)

    for prob, request_id, prefix_len in candidates:
        if prefix_len <= lengths[request_id]:
            continue
        previous_len = int(lengths[request_id])
        delta = float(np.sum(survival[request_id, previous_len:prefix_len]))
        lengths[request_id] = int(prefix_len)
        expected_accepts += delta
        batch_size = int(np.sum(lengths) + base_batch)
        throughput = expected_accepts * _lookup_sps(sps_curve, batch_size)
        is_better = throughput > best_throughput
        if is_better:
            best_throughput = float(throughput)
            best_expected_accepts = float(expected_accepts)
            best_batch_size = int(batch_size)
            best_lengths = lengths.copy()
        trace.append(
            ScheduleStep(
                request_id=int(request_id),
                prefix_len=int(prefix_len),
                survival_probability=float(prob),
                batch_size=int(batch_size),
                expected_accepts=float(expected_accepts),
                throughput=float(throughput),
                accepted=bool(is_better),
            )
        )
        if early_stop and not is_better:
            break

    return PrefixSchedule(
        lengths=best_lengths,
        best_throughput=float(best_throughput),
        expected_accepts=float(best_expected_accepts),
        batch_size=int(best_batch_size),
        trace=tuple(trace),
    )
