from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dspark_opd.config import load_config
from dspark_opd.losses import DSparkSupervisedBatch, OPDBatch, compute_dspark_opd_loss_np
from dspark_opd.scheduler import hardware_aware_prefix_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Run a tiny deterministic DSpark-OPD smoke test.")
    parser.add_argument("--config", default="configs/dspark_opd_qwen3_4b.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    supervised = DSparkSupervisedBatch(
        draft_logits=np.array([[[[3.0, 0.0, -1.0], [0.0, 2.0, -1.0]]]]),
        target_ids=np.array([[[0, 1]]]),
        eval_mask=np.array([[[1.0, 1.0]]]),
        aligned_target_logits=np.array([[[[2.5, 0.2, -0.5], [0.0, 2.5, -1.0]]]]),
        confidence_logits=np.array([[[1.5, 1.0]]]),
    )
    opd = OPDBatch(
        student_logits=np.array([[[3.0, 0.0, -1.0], [0.0, 2.0, -1.0]]]),
        teacher_logits=np.array([[[2.5, 0.2, -0.5], [0.0, 2.5, -1.0]]]),
        mask=np.array([[1.0, 1.0]]),
    )
    loss = compute_dspark_opd_loss_np(supervised, opd, cfg.loss)
    schedule = hardware_aware_prefix_scheduler(
        np.array([[0.9, 0.8, 0.7], [0.5, 0.4, 0.3]]),
        {2: 1.0, 3: 0.95, 4: 0.85, 5: 0.6, 6: 0.4, 7: 0.25, 8: 0.2},
        early_stop=cfg.scheduler.early_stop,
    )
    print(
        json.dumps(
            {
                "project": cfg.project_name,
                "loss": loss.loss,
                "terms": dict(loss.terms),
                "scheduled_lengths": schedule.lengths.tolist(),
                "scheduled_throughput": schedule.best_throughput,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
