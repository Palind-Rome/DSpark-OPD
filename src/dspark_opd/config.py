from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LossConfig:
    """Weights for the combined DSpark + OPD objective."""

    ce_alpha: float = 0.1
    tv_alpha: float = 0.9
    confidence_alpha: float = 1.0
    opd_alpha: float = 1.0
    accepted_forward_kl_alpha: float = 1.0
    rejected_reverse_kl_alpha: float = 1.0
    loss_decay_gamma: float | None = 4.0
    rejected_decay_gamma: float = 0.8
    kl_temperature: float = 1.0
    log_prob_min_clamp: float | None = None
    topk_renormalize_teacher: bool = False


@dataclass(frozen=True)
class SchedulerConfig:
    """Hardware-aware prefix scheduler knobs."""

    early_stop: bool = True
    min_survival_probability: float = 0.0
    include_bonus_token: bool = True


@dataclass(frozen=True)
class TeacherSpec:
    key: str
    model_path: str
    weight: float = 1.0


@dataclass(frozen=True)
class DSparkOPDConfig:
    """Top-level project config.

    The defaults intentionally mirror the public DSpark recipe where possible:
    CE + TV + confidence remain the supervised anchor, and OPD is added as an
    online replay term.
    """

    project_name: str = "dspark-opd"
    target_model_name_or_path: str = "Qwen/Qwen3-4B"
    draft_block_size: int = 7
    replay_block_size: int = 7
    num_replay_anchors: int = 128
    loss: LossConfig = field(default_factory=LossConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    teachers: tuple[TeacherSpec, ...] = field(
        default_factory=lambda: (TeacherSpec(key="default", model_path="Qwen/Qwen3-4B"),)
    )


def _dataclass_from_dict(cls: type, payload: dict[str, Any]):
    field_names = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
    kwargs = {key: value for key, value in payload.items() if key in field_names}
    return cls(**kwargs)


def load_config(path: str | Path) -> DSparkOPDConfig:
    """Load a YAML config into :class:`DSparkOPDConfig`."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw = dict(raw)
    if "loss" in raw:
        raw["loss"] = _dataclass_from_dict(LossConfig, dict(raw["loss"]))
    if "scheduler" in raw:
        raw["scheduler"] = _dataclass_from_dict(SchedulerConfig, dict(raw["scheduler"]))
    if "teachers" in raw:
        raw["teachers"] = tuple(TeacherSpec(**item) for item in raw["teachers"])
    return _dataclass_from_dict(DSparkOPDConfig, raw)
