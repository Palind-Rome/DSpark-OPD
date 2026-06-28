from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .config import LossConfig
from .np_utils import EPS, gather_last, gather_topk, log_softmax, masked_mean, masked_sum, softmax


@dataclass(frozen=True)
class LossBreakdown:
    """Scalar loss plus named metrics."""

    loss: float
    terms: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DSparkSupervisedBatch:
    """Supervised DSpark tensors extracted from a DeepSpec-style forward pass."""

    draft_logits: np.ndarray
    target_ids: np.ndarray
    eval_mask: np.ndarray
    aligned_target_logits: np.ndarray | None = None
    confidence_logits: np.ndarray | None = None


@dataclass(frozen=True)
class OPDBatch:
    """On-policy replay tensors for draft-induced states.

    Shapes:
        student_logits: [..., vocab]
        teacher_logits: [..., vocab], optional full-vocabulary teacher logits
        teacher_topk_ids: [..., topk], optional teacher top-k ids
        teacher_topk_logprobs: [..., topk], optional teacher top-k log-probs
        sampled_token_ids: [...], tokens sampled by the student policy
        mask: [...]
    """

    student_logits: np.ndarray
    teacher_logits: np.ndarray | None = None
    teacher_topk_ids: np.ndarray | None = None
    teacher_topk_logprobs: np.ndarray | None = None
    sampled_token_ids: np.ndarray | None = None
    student_sample_logprobs: np.ndarray | None = None
    teacher_sample_logprobs: np.ndarray | None = None
    mask: np.ndarray | None = None
    accepted_mask: np.ndarray | None = None
    rejected_mask: np.ndarray | None = None
    rejected_weights: np.ndarray | None = None


def _position_weights(shape: tuple[int, ...], gamma: float | None) -> np.ndarray:
    if gamma is None or gamma <= 0:
        return np.ones(shape, dtype=np.float64)
    block_size = int(shape[-1])
    positions = np.arange(block_size, dtype=np.float64)
    weights = np.exp(-positions / float(gamma))
    return np.broadcast_to(weights, shape)


def compute_dspark_supervised_loss_np(
    batch: DSparkSupervisedBatch,
    config: LossConfig | None = None,
) -> LossBreakdown:
    """Compute the public DSpark supervised objective in numpy.

    This mirrors DeepSpec's CE + TV/L1 + confidence-head loss. It is intended for
    small deterministic checks and documentation; production training should use
    the PyTorch adapter.
    """

    cfg = config or LossConfig()
    draft_logits = np.asarray(batch.draft_logits, dtype=np.float64)
    target_ids = np.asarray(batch.target_ids, dtype=np.int64)
    eval_mask = np.asarray(batch.eval_mask, dtype=np.float64)
    weights = eval_mask * _position_weights(eval_mask.shape, cfg.loss_decay_gamma)
    draft_logprobs = log_softmax(draft_logits, axis=-1, temperature=cfg.kl_temperature)
    nll = -gather_last(draft_logprobs, target_ids)
    ce = masked_mean(nll, weights)

    tv = 0.0
    confidence = 0.0
    confidence_abs_error = 0.0
    accept_targets = None
    if batch.aligned_target_logits is not None:
        draft_probs = softmax(draft_logits, axis=-1, temperature=cfg.kl_temperature)
        target_probs = softmax(batch.aligned_target_logits, axis=-1, temperature=cfg.kl_temperature)
        l1 = np.sum(np.abs(draft_probs - target_probs), axis=-1)
        tv = masked_mean(l1, weights)
        accept_targets = np.clip(1.0 - 0.5 * l1, 0.0, 1.0)

    if batch.confidence_logits is not None:
        if accept_targets is None:
            raise ValueError("aligned_target_logits is required for confidence loss.")
        logits = np.asarray(batch.confidence_logits, dtype=np.float64)
        # Stable BCE with soft labels.
        bce = np.maximum(logits, 0.0) - logits * accept_targets + np.log1p(np.exp(-np.abs(logits)))
        confidence = masked_mean(bce, weights)
        confidence_probs = 1.0 / (1.0 + np.exp(-logits))
        confidence_abs_error = masked_mean(np.abs(confidence_probs - accept_targets), weights)

    loss = cfg.ce_alpha * ce + cfg.tv_alpha * tv + cfg.confidence_alpha * confidence
    return LossBreakdown(
        loss=float(loss),
        terms={
            "dspark/ce": float(ce),
            "dspark/tv_l1": float(tv),
            "dspark/confidence": float(confidence),
            "dspark/confidence_abs_error": float(confidence_abs_error),
        },
    )


def compute_reverse_kl_full_np(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    temperature: float = 1.0,
    log_prob_min_clamp: float | None = None,
) -> LossBreakdown:
    """Compute KL(student || teacher) over the full vocabulary."""

    student_logp = log_softmax(student_logits, axis=-1, temperature=temperature)
    teacher_logp = log_softmax(teacher_logits, axis=-1, temperature=temperature)
    if log_prob_min_clamp is not None:
        student_logp = np.maximum(student_logp, float(log_prob_min_clamp))
        teacher_logp = np.maximum(teacher_logp, float(log_prob_min_clamp))
    student_p = np.exp(student_logp)
    per_token = np.sum(student_p * (student_logp - teacher_logp), axis=-1)
    return LossBreakdown(
        loss=masked_mean(per_token, mask),
        terms={
            "opd/reverse_kl_full": masked_mean(per_token, mask),
            "opd/reverse_kl_full_sum": masked_sum(per_token, mask),
        },
    )


def compute_forward_kl_full_np(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    temperature: float = 1.0,
    log_prob_min_clamp: float | None = None,
) -> LossBreakdown:
    """Compute KL(teacher || student) over the full vocabulary."""

    student_logp = log_softmax(student_logits, axis=-1, temperature=temperature)
    teacher_logp = log_softmax(teacher_logits, axis=-1, temperature=temperature)
    if log_prob_min_clamp is not None:
        student_logp = np.maximum(student_logp, float(log_prob_min_clamp))
        teacher_logp = np.maximum(teacher_logp, float(log_prob_min_clamp))
    teacher_p = np.exp(teacher_logp)
    per_token = np.sum(teacher_p * (teacher_logp - student_logp), axis=-1)
    return LossBreakdown(
        loss=masked_mean(per_token, mask),
        terms={
            "opd/forward_kl_full": masked_mean(per_token, mask),
            "opd/forward_kl_full_sum": masked_sum(per_token, mask),
        },
    )


def compute_forward_kl_topk_np(
    student_logits: np.ndarray,
    teacher_topk_ids: np.ndarray,
    teacher_topk_logprobs: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    temperature: float = 1.0,
    renormalize_teacher: bool = False,
    log_prob_min_clamp: float | None = None,
) -> LossBreakdown:
    """Compute teacher-top-k forward KL approximation.

    This matches the practical GKD route used in verl: the teacher supplies
    only top-k ids/log-probs, and the student gathers log-probs at those ids.
    """

    student_logp = log_softmax(student_logits, axis=-1, temperature=temperature)
    student_topk_logp = gather_topk(student_logp, teacher_topk_ids)
    teacher_logp = np.asarray(teacher_topk_logprobs, dtype=np.float64)
    if renormalize_teacher:
        teacher_logp = teacher_logp - np.log(np.sum(np.exp(teacher_logp), axis=-1, keepdims=True))
    if log_prob_min_clamp is not None:
        student_topk_logp = np.maximum(student_topk_logp, float(log_prob_min_clamp))
        teacher_logp = np.maximum(teacher_logp, float(log_prob_min_clamp))
    teacher_p = np.exp(teacher_logp)
    per_token = np.sum(teacher_p * (teacher_logp - student_topk_logp), axis=-1)
    teacher_mass = np.sum(np.exp(np.asarray(teacher_topk_logprobs, dtype=np.float64)), axis=-1)
    student_mass = np.sum(np.exp(student_topk_logp), axis=-1)
    return LossBreakdown(
        loss=masked_mean(per_token, mask),
        terms={
            "opd/forward_kl_topk": masked_mean(per_token, mask),
            "opd/teacher_topk_mass": masked_mean(teacher_mass, mask),
            "opd/student_mass_on_teacher_topk": masked_mean(student_mass, mask),
        },
    )


def compute_single_sample_kl_estimator_np(
    student_logprobs: np.ndarray,
    teacher_logprobs: np.ndarray,
    *,
    mode: str = "k1",
    mask: np.ndarray | None = None,
) -> LossBreakdown:
    """Single-sample reverse-KL estimators used by PG OPD.

    The returned value is a per-token divergence estimate. In a policy-gradient
    update it should be detached and used as a negative reward/advantage.
    """

    log_ratio = np.asarray(student_logprobs, dtype=np.float64) - np.asarray(
        teacher_logprobs, dtype=np.float64
    )
    mode = mode.lower()
    if mode in {"kl", "k1"}:
        per_token = log_ratio
    elif mode == "abs":
        per_token = np.abs(log_ratio)
    elif mode == "mse":
        per_token = log_ratio**2
    elif mode == "k2":
        per_token = 0.5 * log_ratio**2
    elif mode in {"low_var_kl", "k3"}:
        # Schulman k3 estimator: exp(log q - log p) - 1 + log p - log q.
        ratio_inv = np.exp(-log_ratio)
        per_token = ratio_inv - 1.0 + log_ratio
    else:
        raise ValueError(f"Unsupported estimator mode: {mode}")
    return LossBreakdown(
        loss=masked_mean(per_token, mask),
        terms={
            f"opd/{mode}": masked_mean(per_token, mask),
            f"opd/{mode}_abs": masked_mean(np.abs(per_token), mask),
        },
    )


def _require_opd_teacher(batch: OPDBatch) -> None:
    has_full = batch.teacher_logits is not None
    has_topk = batch.teacher_topk_ids is not None and batch.teacher_topk_logprobs is not None
    has_sample = batch.student_sample_logprobs is not None and batch.teacher_sample_logprobs is not None
    if not (has_full or has_topk or has_sample):
        raise ValueError("OPDBatch needs full logits, top-k teacher logprobs, or sampled logprobs.")


def compute_opd_loss_np(batch: OPDBatch, config: LossConfig | None = None) -> LossBreakdown:
    """Compute the OPD replay loss.

    If accepted/rejected masks are provided, this follows the Draft-OPD idea:
    accepted tokens use forward KL, rejected tokens use reverse KL. Without
    those masks, the function falls back to one aggregate OPD term.
    """

    cfg = config or LossConfig()
    _require_opd_teacher(batch)
    mask = None if batch.mask is None else np.asarray(batch.mask, dtype=np.float64)

    accepted_mask = None
    if batch.accepted_mask is not None:
        accepted_mask = np.asarray(batch.accepted_mask, dtype=np.float64)
        accepted_mask = accepted_mask if mask is None else accepted_mask * mask
    rejected_mask = None
    if batch.rejected_mask is not None:
        rejected_mask = np.asarray(batch.rejected_mask, dtype=np.float64)
        rejected_mask = rejected_mask if mask is None else rejected_mask * mask
        if batch.rejected_weights is not None:
            rejected_mask = rejected_mask * np.asarray(batch.rejected_weights, dtype=np.float64)

    terms: dict[str, float] = {}
    loss = 0.0

    if accepted_mask is not None and rejected_mask is not None:
        if batch.teacher_logits is not None:
            accepted = compute_forward_kl_full_np(
                batch.student_logits,
                batch.teacher_logits,
                mask=accepted_mask,
                temperature=cfg.kl_temperature,
                log_prob_min_clamp=cfg.log_prob_min_clamp,
            )
            rejected = compute_reverse_kl_full_np(
                batch.student_logits,
                batch.teacher_logits,
                mask=rejected_mask,
                temperature=cfg.kl_temperature,
                log_prob_min_clamp=cfg.log_prob_min_clamp,
            )
        else:
            accepted = compute_forward_kl_topk_np(
                batch.student_logits,
                batch.teacher_topk_ids,
                batch.teacher_topk_logprobs,
                mask=accepted_mask,
                temperature=cfg.kl_temperature,
                renormalize_teacher=cfg.topk_renormalize_teacher,
                log_prob_min_clamp=cfg.log_prob_min_clamp,
            )
            if batch.student_sample_logprobs is None or batch.teacher_sample_logprobs is None:
                raise ValueError(
                    "Rejected-token reverse KL needs full teacher logits or sampled "
                    "student/teacher logprobs. Teacher top-k alone only supports the "
                    "accepted-token forward-KL branch."
                )
            else:
                student_sample = batch.student_sample_logprobs
                teacher_sample = batch.teacher_sample_logprobs
            rejected = compute_single_sample_kl_estimator_np(
                student_sample,
                teacher_sample,
                mode="k3",
                mask=rejected_mask,
            )
        loss = (
            cfg.accepted_forward_kl_alpha * accepted.loss
            + cfg.rejected_reverse_kl_alpha * rejected.loss
        ) / max(cfg.accepted_forward_kl_alpha + cfg.rejected_reverse_kl_alpha, EPS)
        terms.update(accepted.terms)
        terms.update(rejected.terms)
        terms["opd/accepted_tokens"] = masked_sum(np.ones_like(accepted_mask), accepted_mask)
        terms["opd/rejected_tokens"] = masked_sum(np.ones_like(rejected_mask), rejected_mask)
    elif batch.teacher_logits is not None:
        aggregate = compute_reverse_kl_full_np(
            batch.student_logits,
            batch.teacher_logits,
            mask=mask,
            temperature=cfg.kl_temperature,
            log_prob_min_clamp=cfg.log_prob_min_clamp,
        )
        loss = aggregate.loss
        terms.update(aggregate.terms)
    elif batch.teacher_topk_ids is not None and batch.teacher_topk_logprobs is not None:
        aggregate = compute_forward_kl_topk_np(
            batch.student_logits,
            batch.teacher_topk_ids,
            batch.teacher_topk_logprobs,
            mask=mask,
            temperature=cfg.kl_temperature,
            renormalize_teacher=cfg.topk_renormalize_teacher,
            log_prob_min_clamp=cfg.log_prob_min_clamp,
        )
        loss = aggregate.loss
        terms.update(aggregate.terms)
    else:
        aggregate = compute_single_sample_kl_estimator_np(
            batch.student_sample_logprobs,
            batch.teacher_sample_logprobs,
            mode="k3",
            mask=mask,
        )
        loss = aggregate.loss
        terms.update(aggregate.terms)

    terms["opd/loss"] = float(loss)
    return LossBreakdown(loss=float(loss), terms=terms)


def compute_dspark_opd_loss_np(
    supervised: DSparkSupervisedBatch,
    opd: OPDBatch | None,
    config: LossConfig | None = None,
) -> LossBreakdown:
    """Combine DSpark supervised training with an OPD replay term."""

    cfg = config or LossConfig()
    supervised_loss = compute_dspark_supervised_loss_np(supervised, cfg)
    terms = dict(supervised_loss.terms)
    total = supervised_loss.loss
    if opd is not None and cfg.opd_alpha > 0:
        opd_loss = compute_opd_loss_np(opd, cfg)
        total += cfg.opd_alpha * opd_loss.loss
        terms.update(opd_loss.terms)
    terms["loss/total"] = float(total)
    return LossBreakdown(loss=float(total), terms=terms)
