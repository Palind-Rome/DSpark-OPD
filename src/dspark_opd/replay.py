from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReplayBlock:
    """One draft block collected during speculative rollout.

    ``accepted_count`` is the number of draft tokens accepted by target
    verification. The block can then be replayed to compute teacher/student
    log-probs on both accepted and rejected draft-induced states.
    """

    request_id: int
    anchor_position: int
    draft_token_ids: np.ndarray
    accepted_count: int


def split_acceptance_masks(accepted_counts: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Build accepted/rejected masks from per-block accepted prefix lengths."""

    counts = np.asarray(accepted_counts, dtype=np.int64)
    positions = np.arange(int(block_size), dtype=np.int64)
    accepted = positions.reshape((1,) * counts.ndim + (block_size,)) < np.expand_dims(counts, -1)
    valid = positions.reshape((1,) * counts.ndim + (block_size,)) < int(block_size)
    rejected = valid & ~accepted
    return accepted.astype(np.float64), rejected.astype(np.float64)


def position_decay_weights(block_size: int, gamma: float = 0.8) -> np.ndarray:
    """Exponentially decay rejected-token weights by draft position."""

    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    positions = np.arange(int(block_size), dtype=np.float64)
    return np.power(float(gamma), positions)


def replay_blocks_to_masks(
    replay_blocks: list[ReplayBlock],
    *,
    block_size: int,
    rejected_gamma: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create masks aligned to a flat list of replay blocks."""

    counts = np.asarray([block.accepted_count for block in replay_blocks], dtype=np.int64)
    accepted, rejected = split_acceptance_masks(counts, block_size)
    weights = np.broadcast_to(position_decay_weights(block_size, rejected_gamma), rejected.shape)
    return accepted, rejected, weights
