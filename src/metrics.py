"""Evaluador de detección estilo COCO con soporte de cajas *ignoradas*.

Se implementa aquí (en lugar de usar solo `model.val()` de Ultralytics) porque los
análisis de robustez necesitan evaluar subgrupos de cajas (por tamaño u oclusión):
las cajas fuera del subgrupo se marcan como `ignore`, de modo que una detección que
las encuentra no cuenta ni como TP ni como FP (misma lógica que COCOeval con
rangos de área). AP con interpolación de 101 puntos, IoU 0.50:0.05:0.95.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

IOU_THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)
REC_PTS = np.linspace(0, 1, 101)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre cajas xyxy: a (D,4), b (G,4) -> (D,G)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def _match(ious: np.ndarray, g_ign: np.ndarray, thr: float):
    """Emparejamiento greedy COCO para un par (imagen, clase) ya ordenado por score."""
    D, G = ious.shape
    tp = np.zeros(D, bool)
    ign = np.zeros(D, bool)
    g_used = np.zeros(G, bool)
    order = np.argsort(g_ign, kind="stable")  # GT no ignoradas primero
    for d in range(D):
        best, m = min(thr, 1 - 1e-10), -1
        for g in order:
            if g_used[g]:
                continue
            if m > -1 and not g_ign[m] and g_ign[g]:
                break
            if ious[d, g] < best:
                continue
            best, m = ious[d, g], g
        if m == -1:
            continue
        g_used[m] = True
        if g_ign[m]:
            ign[d] = True
        else:
            tp[d] = True
    return tp, ign


def _ap(scores, tp, ign, npos) -> float:
    if npos == 0:
        return np.nan
    keep = ~ign
    s, t = scores[keep], tp[keep]
    if len(s) == 0:
        return 0.0
    o = np.argsort(-s, kind="mergesort")
    t = t[o]
    ctp, cfp = np.cumsum(t), np.cumsum(~t)
    rec = ctp / npos
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]  # envolvente
    idx = np.searchsorted(rec, REC_PTS, side="left")
    q = np.where(idx < len(prec), prec[np.minimum(idx, len(prec) - 1)], 0.0)
    return float(q.mean())


def evaluate(gt: pd.DataFrame, det: pd.DataFrame, image_ids, classes: list[str]) -> pd.DataFrame:
    """gt: image_id, category, x1,y1,x2,y2, [ignore]; det: image_id, category, x1..y2, score.

    Devuelve un DataFrame por clase con AP50, AP75, AP50:95, n_gt.
    """
    image_ids = set(map(str, image_ids))
    gt = gt[gt["image_id"].isin(image_ids)].copy()
    det = det[det["image_id"].isin(image_ids)].copy()
    if "ignore" not in gt.columns:
        gt["ignore"] = False
    gt["ignore"] = gt["ignore"].astype(bool)
    cols = ["x1", "y1", "x2", "y2"]
    acc = {c: {t: ([], [], []) for t in IOU_THRS} for c in classes}
    det = det.sort_values("score", ascending=False)
    g_groups = {k: v for k, v in gt.groupby(["image_id", "category"])}
    for (iid, cat), d in det.groupby(["image_id", "category"], sort=False):
        if cat not in acc:
            continue
        g = g_groups.get((iid, cat))
        sc = d["score"].to_numpy()
        if g is None:
            for t in IOU_THRS:
                acc[cat][t][0].append(sc)
                acc[cat][t][1].append(np.zeros(len(d), bool))
                acc[cat][t][2].append(np.zeros(len(d), bool))
            continue
        ious = iou_matrix(d[cols].to_numpy(float), g[cols].to_numpy(float))
        g_ign = g["ignore"].to_numpy()
        for t in IOU_THRS:
            tp, ig = _match(ious, g_ign, t)
            acc[cat][t][0].append(sc)
            acc[cat][t][1].append(tp)
            acc[cat][t][2].append(ig)
    rows = []
    for c in classes:
        npos = int((~gt[gt["category"] == c]["ignore"]).sum())
        aps = []
        for t in IOU_THRS:
            s, tp, ig = acc[c][t]
            if s:
                aps.append(_ap(np.concatenate(s), np.concatenate(tp), np.concatenate(ig), npos))
            else:
                aps.append(np.nan if npos == 0 else 0.0)
        aps = np.array(aps, float)
        rows.append(dict(category=c, n_gt=npos, AP50=aps[0], AP75=aps[5],
                         AP50_95=np.nan if np.isnan(aps).all() else np.nanmean(aps)))
    return pd.DataFrame(rows)


def summarize(per_class: pd.DataFrame) -> dict:
    v = per_class[per_class["n_gt"] > 0]
    return {"mAP50": float(v["AP50"].mean()), "mAP50_95": float(v["AP50_95"].mean()),
            "n_classes": int(len(v)), "n_gt": int(v["n_gt"].sum())}


def image_level(meta: pd.DataFrame, det: pd.DataFrame, thr: float) -> dict:
    """Métricas de filtrado de fotos vacías: '¿hay algún animal?' por imagen."""
    mx = det.groupby("image_id")["score"].max()
    y = (meta["status"] == "animal").to_numpy()
    s = meta["image_id"].map(mx).fillna(0).to_numpy()
    pred = s >= thr
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    out = dict(threshold=thr, precision=prec, recall=rec,
               f1=2 * prec * rec / max(prec + rec, 1e-9),
               empty_fp_rate=fp / max(fp + tn, 1), tp=tp, fp=fp, fn=fn, tn=tn)
    try:
        from sklearn.metrics import roc_auc_score
        out["auc"] = float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")
    except ImportError:
        pass
    return out


def best_threshold(meta: pd.DataFrame, det: pd.DataFrame) -> float:
    grid = np.round(np.arange(0.05, 0.95, 0.025), 3)
    f1 = [image_level(meta, det, t)["f1"] for t in grid]
    return float(grid[int(np.argmax(f1))])
