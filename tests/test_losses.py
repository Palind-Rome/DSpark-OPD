import numpy as np

from dspark_opd.losses import (
    DSparkSupervisedBatch,
    OPDBatch,
    compute_dspark_opd_loss_np,
    compute_dspark_supervised_loss_np,
    compute_forward_kl_topk_np,
    compute_reverse_kl_full_np,
)


def test_reverse_kl_is_zero_for_identical_logits():
    logits = np.array([[[1.0, 0.0, -1.0], [0.0, 2.0, -0.5]]])
    mask = np.array([[1.0, 1.0]])
    out = compute_reverse_kl_full_np(logits, logits, mask=mask)
    assert abs(out.loss) < 1e-10


def test_forward_kl_topk_matches_full_when_topk_is_full_vocab():
    student = np.array([[[2.0, 0.0, -1.0]]])
    teacher = np.array([[[1.5, 0.5, -0.5]]])
    teacher_logp = teacher - np.log(np.exp(teacher).sum(axis=-1, keepdims=True))
    ids = np.array([[[0, 1, 2]]])
    topk = compute_forward_kl_topk_np(student, ids, teacher_logp, mask=np.array([[1.0]]))

    from dspark_opd.losses import compute_forward_kl_full_np

    full = compute_forward_kl_full_np(student, teacher, mask=np.array([[1.0]]))
    assert np.allclose(topk.loss, full.loss)


def test_dspark_supervised_loss_combines_terms():
    draft = np.array([[[[3.0, 0.0], [0.0, 2.0]]]])
    target = np.array([[[0, 1]]])
    mask = np.array([[[1.0, 1.0]]])
    confidence_logits = np.array([[[1.0, 1.0]]])
    batch = DSparkSupervisedBatch(
        draft_logits=draft,
        target_ids=target,
        eval_mask=mask,
        aligned_target_logits=draft.copy(),
        confidence_logits=confidence_logits,
    )
    out = compute_dspark_supervised_loss_np(batch)
    assert out.loss > 0
    assert out.terms["dspark/tv_l1"] == 0.0
    assert out.terms["dspark/confidence"] > 0


def test_combined_loss_adds_opd_term():
    supervised = DSparkSupervisedBatch(
        draft_logits=np.array([[[[3.0, 0.0], [0.0, 2.0]]]]),
        target_ids=np.array([[[0, 1]]]),
        eval_mask=np.array([[[1.0, 1.0]]]),
        aligned_target_logits=np.array([[[[3.0, 0.0], [0.0, 2.0]]]]),
    )
    opd = OPDBatch(
        student_logits=np.array([[[3.0, 0.0], [0.0, 2.0]]]),
        teacher_logits=np.array([[[2.0, 1.0], [1.0, 2.0]]]),
        mask=np.array([[1.0, 1.0]]),
    )
    supervised_only = compute_dspark_opd_loss_np(supervised, None)
    combined = compute_dspark_opd_loss_np(supervised, opd)
    assert combined.loss > supervised_only.loss
    assert "opd/reverse_kl_full" in combined.terms
