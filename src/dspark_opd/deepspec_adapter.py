from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepSpecIntegrationPlan:
    """Concrete patch points for integrating OPD into DeepSpec DSpark."""

    trainer_file: str = "deepspec/trainer/dspark_trainer.py"
    loss_file: str = "deepspec/modeling/dspark/loss.py"
    data_file: str = "deepspec/data/target_cache_dataset.py"
    summary: str = (
        "Add an online replay producer next to the target-cache batch, replay "
        "DSpark draft anchors against the frozen target teacher, then call "
        "dspark_opd.torch_losses.compute_dspark_opd_torch_loss from run_batch."
    )


def integration_plan() -> DeepSpecIntegrationPlan:
    """Return the minimal DeepSpec patch plan documented by this project."""

    return DeepSpecIntegrationPlan()
