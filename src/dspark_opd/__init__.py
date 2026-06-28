"""DSpark-OPD research engineering toolkit.

The package keeps the numerically testable pieces lightweight and dependency
small. PyTorch integration lives in :mod:`dspark_opd.torch_losses` and is only
imported by users who have torch installed.
"""

from .calibration import CalibrationResult, expected_calibration_error, sequential_temperature_scaling
from .config import DSparkOPDConfig, LossConfig, SchedulerConfig, load_config
from .losses import (
    DSparkSupervisedBatch,
    LossBreakdown,
    OPDBatch,
    compute_dspark_opd_loss_np,
    compute_dspark_supervised_loss_np,
    compute_forward_kl_topk_np,
    compute_reverse_kl_full_np,
    compute_single_sample_kl_estimator_np,
)
from .replay import ReplayBlock, split_acceptance_masks, position_decay_weights
from .scheduler import PrefixSchedule, hardware_aware_prefix_scheduler

__all__ = [
    "CalibrationResult",
    "DSparkOPDConfig",
    "DSparkSupervisedBatch",
    "LossBreakdown",
    "LossConfig",
    "OPDBatch",
    "PrefixSchedule",
    "ReplayBlock",
    "SchedulerConfig",
    "compute_dspark_opd_loss_np",
    "compute_dspark_supervised_loss_np",
    "compute_forward_kl_topk_np",
    "compute_reverse_kl_full_np",
    "compute_single_sample_kl_estimator_np",
    "expected_calibration_error",
    "hardware_aware_prefix_scheduler",
    "load_config",
    "position_decay_weights",
    "sequential_temperature_scaling",
    "split_acceptance_masks",
]
