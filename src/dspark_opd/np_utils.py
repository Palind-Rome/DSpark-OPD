from __future__ import annotations

import numpy as np


EPS = 1e-12


def softmax(logits: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    shifted = scaled - np.max(scaled, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def log_softmax(logits: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    shifted = scaled - np.max(scaled, axis=axis, keepdims=True)
    log_z = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - log_z


def gather_last(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    indices = np.asarray(indices, dtype=np.int64)
    expanded = np.expand_dims(indices, axis=-1)
    gathered = np.take_along_axis(values, expanded, axis=-1)
    return np.squeeze(gathered, axis=-1)


def gather_topk(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    indices = np.asarray(indices, dtype=np.int64)
    return np.take_along_axis(values, indices, axis=-1)


def masked_sum(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=np.float64)
    if mask is None:
        return float(np.sum(values))
    return float(np.sum(values * np.asarray(mask, dtype=np.float64)))


def masked_mean(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=np.float64)
    if mask is None:
        return float(np.mean(values))
    mask_f = np.asarray(mask, dtype=np.float64)
    den = float(np.sum(mask_f))
    if den <= 0.0:
        return 0.0
    return float(np.sum(values * mask_f) / den)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x_arr)
    pos = x_arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    exp_x = np.exp(x_arr[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    if np.isscalar(x):
        return float(out)
    return out


def logit(p: np.ndarray | float, eps: float = EPS) -> np.ndarray | float:
    p_arr = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    out = np.log(p_arr / (1.0 - p_arr))
    if np.isscalar(p):
        return float(out)
    return out
