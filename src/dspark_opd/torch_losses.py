from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TorchLossConfig:
    ce_alpha: float = 0.1
    tv_alpha: float = 0.9
    confidence_alpha: float = 1.0
    opd_alpha: float = 1.0
    loss_decay_gamma: float | None = 4.0
    topk_renormalize_teacher: bool = False
    log_prob_min_clamp: float | None = None


def _import_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
        raise ModuleNotFoundError(
            "dspark_opd.torch_losses requires PyTorch. Install the package with "
            "`pip install dspark-opd[torch]` or run inside the DeepSpec environment."
        ) from exc
    return torch, F


def _build_weight_mask(eval_mask, gamma: float | None):
    torch, _ = _import_torch()
    weights = eval_mask.to(dtype=torch.float32)
    if gamma is not None and gamma > 0:
        block_size = eval_mask.shape[-1]
        positions = torch.arange(block_size, device=eval_mask.device, dtype=torch.float32)
        weights = weights * torch.exp(-positions / float(gamma))
    return weights


def compute_forward_kl_topk(student_logits, teacher_topk_ids, teacher_topk_logprobs, mask=None, config=None):
    """Memory-conscious teacher-top-k forward KL for PyTorch tensors."""

    torch, F = _import_torch()
    cfg = config or TorchLossConfig()
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    student_logp = torch.gather(student_logp, dim=-1, index=teacher_topk_ids.long())
    teacher_logp = teacher_topk_logprobs.float()
    if cfg.topk_renormalize_teacher:
        teacher_logp = teacher_logp - torch.logsumexp(teacher_logp, dim=-1, keepdim=True)
    if cfg.log_prob_min_clamp is not None:
        student_logp = student_logp.clamp_min(float(cfg.log_prob_min_clamp))
        teacher_logp = teacher_logp.clamp_min(float(cfg.log_prob_min_clamp))
    per_token = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1)
    if mask is None:
        return per_token.mean()
    mask_f = mask.float()
    return (per_token * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def compute_reverse_kl_full(student_logits, teacher_logits, mask=None, config=None):
    torch, F = _import_torch()
    del torch
    cfg = config or TorchLossConfig()
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    if cfg.log_prob_min_clamp is not None:
        student_logp = student_logp.clamp_min(float(cfg.log_prob_min_clamp))
        teacher_logp = teacher_logp.clamp_min(float(cfg.log_prob_min_clamp))
    per_token = (student_logp.exp() * (student_logp - teacher_logp)).sum(dim=-1)
    if mask is None:
        return per_token.mean()
    mask_f = mask.float()
    return (per_token * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def compute_dspark_supervised_loss_from_outputs(outputs: Any, config: TorchLossConfig | None = None):
    """DeepSpec-compatible DSpark supervised loss.

    ``outputs`` is duck-typed to DeepSpec's ``DSparkForwardOutput``.
    """

    torch, F = _import_torch()
    cfg = config or TorchLossConfig()
    draft_logits = outputs.draft_logits
    target_ids = outputs.target_ids
    eval_mask = outputs.eval_mask
    weights = _build_weight_mask(eval_mask, cfg.loss_decay_gamma)
    ce_per_token = F.cross_entropy(
        draft_logits.reshape(-1, draft_logits.shape[-1]).float(),
        target_ids.reshape(-1).long(),
        reduction="none",
    ).reshape_as(target_ids)
    ce = (ce_per_token * weights).sum() / weights.sum().clamp_min(1.0)

    tv = draft_logits.new_zeros((), dtype=torch.float32)
    accept_targets = None
    if getattr(outputs, "aligned_target_logits", None) is not None:
        draft_probs = F.softmax(draft_logits.float(), dim=-1)
        target_probs = F.softmax(outputs.aligned_target_logits.float(), dim=-1)
        l1 = (draft_probs - target_probs).abs().sum(dim=-1)
        tv = (l1 * weights).sum() / weights.sum().clamp_min(1.0)
        accept_targets = (1.0 - 0.5 * l1).clamp(0.0, 1.0).detach()

    conf = draft_logits.new_zeros((), dtype=torch.float32)
    if getattr(outputs, "confidence_pred", None) is not None:
        if accept_targets is None:
            raise ValueError("aligned_target_logits is required for confidence loss.")
        bce = F.binary_cross_entropy_with_logits(
            outputs.confidence_pred.float(),
            accept_targets,
            reduction="none",
        )
        conf = (bce * weights).sum() / weights.sum().clamp_min(1.0)

    return cfg.ce_alpha * ce + cfg.tv_alpha * tv + cfg.confidence_alpha * conf


def compute_dspark_opd_torch_loss(
    outputs: Any,
    *,
    teacher_logits=None,
    teacher_topk_ids=None,
    teacher_topk_logprobs=None,
    replay_mask=None,
    config: TorchLossConfig | None = None,
):
    """Combine DeepSpec DSpark outputs with an OPD replay term.

    In a real DeepSpec trainer, ``outputs`` comes from the supervised target-cache
    batch, while the teacher tensors come from replaying draft-induced anchors.
    """

    cfg = config or TorchLossConfig()
    loss = compute_dspark_supervised_loss_from_outputs(outputs, cfg)
    if cfg.opd_alpha <= 0:
        return loss
    mask = outputs.eval_mask if replay_mask is None else replay_mask
    if teacher_logits is not None:
        opd = compute_reverse_kl_full(outputs.draft_logits, teacher_logits, mask, cfg)
    elif teacher_topk_ids is not None and teacher_topk_logprobs is not None:
        opd = compute_forward_kl_topk(
            outputs.draft_logits,
            teacher_topk_ids,
            teacher_topk_logprobs,
            mask,
            cfg,
        )
    else:
        return loss
    return loss + cfg.opd_alpha * opd
