from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dspark_opd.losses import DSparkSupervisedBatch, OPDBatch, compute_dspark_opd_loss_np


supervised = DSparkSupervisedBatch(
    draft_logits=np.array([[[[2.0, 0.0], [0.0, 2.0]]]]),
    target_ids=np.array([[[0, 1]]]),
    eval_mask=np.array([[[1.0, 1.0]]]),
    aligned_target_logits=np.array([[[[1.5, 0.2], [0.1, 2.2]]]]),
)

opd = OPDBatch(
    student_logits=np.array([[[2.0, 0.0], [0.0, 2.0]]]),
    teacher_logits=np.array([[[1.5, 0.2], [0.1, 2.2]]]),
    mask=np.array([[1.0, 1.0]]),
)

print(compute_dspark_opd_loss_np(supervised, opd))
