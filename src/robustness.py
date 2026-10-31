"""Robustez ante oclusión y tamaño del objeto (entregable 4).

CCT-20 no trae etiquetas de oclusión, así que se usan dos fuentes:
  1. Proxy automático para TODAS las cajas de test:
       score = 0.5·truncamiento (caja cortada por el borde de la imagen)
             + 0.5·rango percentil de la densidad de bordes (Canny) dentro de la caja
     (la vegetación delante del animal fragmenta su silueta y eleva la densidad de
     bordes). Se discretiza en terciles: bajo / medio / alto.
  2. Anotación manual de una muestra estratificada (--make-sheet genera los
     recortes y una plantilla CSV). Si existe results/occlusion_manual.csv se
     evalúa también sobre esas etiquetas y se reporta el acuerdo (kappa ponderado)
     entre proxy y anotación humana — esto valida (o invalida) el proxy.

El tamaño relativo es área_caja / área_imagen con cortes en configs (1 % y 10 %).
Para cada grupo se calcula AP marcando como `ignore` las cajas de otros grupos.

Uso:  python src/robustness.py [--exp baseline] [--make-sheet]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_config, p, save_json, to_latex
from metrics import evaluate, iou_matrix, summarize
from predict import gt_frame, predict_split, read_list


def edge_density(path: str, x: float, y: float, w: float, h: float) -> float:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.nan
    x0, y0 = int(max(0, x)), int(max(0, y))
    crop = img[y0:int(y0 + h), x0:int(x0 + w)]
    if crop.size < 16:
        return np.nan
    crop = cv2.GaussianBlur(crop, (3, 3), 0)
    med = float(np.median(crop))
    edges = cv2.Canny(crop, int(max(0, 0.66 * med)), int(min(255, 1.33 * med) + 1))
    return float((edges > 0).mean())


def add_proxies(cfg, gt: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    cache = p(cfg, "results") / "occlusion_proxy.csv"
    if cache.exists():
        c = pd.read_csv(cache)
        if set(gt["box_id"]).issubset(set(c["box_id"])):
            return gt.merge(c[["box_id", "edge_density"]], on="box_id", how="left")
    paths = meta.set_index("image_id")["path"]
    gt = gt.copy()
    gt["edge_density"] = [edge_density(paths[i], x, y, w, h) for i, x, y, w, h in
                          zip(gt["image_id"], gt["x"], gt["y"], gt["w"], gt["h"])]
    gt[["box_id", "edge_density"]].to_csv(cache, index=False)
    return gt


def recall_at(gt: pd.DataFrame, det: pd.DataFrame, thr: float, iou: float = 0.5) -> pd.Series:
    """Para cada caja GT: ¿la encuentra una detección de su clase con score>=thr e IoU>=iou?"""
    d = det[det["score"] >= thr]
    dg = {k: v for k, v in d.groupby(["image_id", "category"])}
    hit = np.zeros(len(gt), bool)
    cols = ["x1", "y1", "x2", "y2"]
    for j, ((iid, cat), g) in enumerate(gt.groupby(["image_id", "category"])):
        dd = dg.get((iid, cat))
        if dd is None:
            continue
        m = iou_matrix(g[cols].to_numpy(float), dd[cols].to_numpy(float)).max(1) >= iou
        hit[gt.index.get_indexer(g.index)] = m
    return pd.Series(hit, index=gt.index)


def group_eval(gt, det, image_ids, classes, col, labels, thr):
    rows = []
    hits = recall_at(gt, det, thr)
    for lab in labels:
        g = gt.copy()
        g["ignore"] = g[col] != lab
        s = summarize(evaluate(g, det, image_ids, classes))
        rows.append({"grupo": lab, "cajas": int((~g["ignore"]).sum()),
                     "mAP50": s["mAP50"], "mAP50:95": s["mAP50_95"],
                     "recall": float(hits[gt[col] == lab].mean()) if (gt[col] == lab).any() else np.nan})
    df = pd.DataFrame(rows)
    df["caída mAP50"] = df["mAP50"].max() - df["mAP50"]
    return df


def make_sheet(cfg, gt, meta, n):
    out = p(cfg, "results") / "occlusion_sheet"
    out.mkdir(parents=True, exist_ok=True)
    s = gt.groupby("occ_proxy", group_keys=False).apply(
        lambda g: g.sample(min(len(g), n // 3), random_state=cfg["seed"]))
    s = s.sample(frac=1, random_state=cfg["seed"])  # barajar: el anotador no ve el proxy
    paths = meta.set_index("image_id")["path"]
    html = ["<html><body style='font-family:sans-serif'><h3>Oclusión: bajo / medio / alto</h3>"]
    for r in s.itertuples():
        img = cv2.imread(paths[r.image_id])
        pad = 0.25
        x0, y0 = int(max(0, r.x - pad * r.w)), int(max(0, r.y - pad * r.h))
        x1, y1 = int(min(img.shape[1], r.x + (1 + pad) * r.w)), int(min(img.shape[0], r.y + (1 + pad) * r.h))
        crop = img[y0:y1, x0:x1].copy()
        cv2.rectangle(crop, (int(r.x - x0), int(r.y - y0)), (int(r.x - x0 + r.w), int(r.y - y0 + r.h)), (0, 255, 255), 2)
        fn = f"box_{r.box_id}.jpg"
        cv2.imwrite(str(out / fn), crop)
        html.append(f"<div style='display:inline-block;margin:6px;text-align:center'>"
                    f"<img src='{fn}' height=180><br>box_id={r.box_id} ({r.category})</div>")
    (out / "index.html").write_text("\n".join(html) + "</body></html>")
    s[["box_id", "image_id", "category"]].assign(occlusion="").to_csv(
        p(cfg, "results") / "occlusion_manual_TEMPLATE.csv", index=False)
    print(f"[hoja] {len(s)} recortes en {out}. Completa la columna 'occlusion' con bajo/medio/alto "
          f"y guarda como results/occlusion_manual.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="baseline")
    ap.add_argument("--make-sheet", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    rcfg = cfg["robustness"]
    res, tab, fig = p(cfg, "results"), p(cfg, "report_tables"), p(cfg, "report_figures")
    meta = pd.read_csv(res / "metadata.csv", dtype={"image_id": str})
    boxes = pd.read_csv(res / "boxes.csv", dtype={"image_id": str})
    boxes["box_id"] = np.arange(len(boxes))
    classes = json.loads((p(cfg, "yolo") / "classes.json").read_text())
    thr = json.loads((res / "operating_threshold.json").read_text())["threshold"] \
        if (res / "operating_threshold.json").exists() else cfg["eval"]["operating_conf"]
    w = p(cfg, "runs") / args.exp / "weights" / "best.pt"

    out, tables = {}, []
    for split in ["cis_test", "trans_test"]:
        det = predict_split(w, split, args.exp, cfg)
        gt = gt_frame(boxes, split)
        ids = [Path(f).stem for f in read_list(p(cfg, "yolo") / "lists" / f"{split}.txt")]
        gt = gt[gt["image_id"].isin(set(ids))]
        gt["size"] = pd.cut(gt["rel_area"], rcfg["size_bins"], labels=rcfg["size_labels"],
                            include_lowest=True).astype(str)
        gt = add_proxies(cfg, gt, meta)
        score = 0.5 * gt["border"] + 0.5 * gt["edge_density"].rank(pct=True).fillna(0.5)
        gt["occ_proxy"] = pd.qcut(score.rank(method="first"), 3, labels=rcfg["occlusion_labels"]).astype(str)

        if args.make_sheet and split == "trans_test":
            make_sheet(cfg, gt, meta, rcfg["manual_sample"])

        for col, labels, name in [("size", rcfg["size_labels"], "tamaño"),
                                  ("occ_proxy", rcfg["occlusion_labels"], "oclusión (proxy)")]:
            df = group_eval(gt, det, ids, classes, col, labels, thr)
            df.insert(0, "split", split.replace("_test", ""))
            df.insert(1, "factor", name)
            tables.append(df)

        man_f = res / "occlusion_manual.csv"
        if man_f.exists() and split == "trans_test":
            man = pd.read_csv(man_f).dropna(subset=["occlusion"])
            man = man[man["occlusion"].isin(rcfg["occlusion_labels"])]
            g2 = gt.merge(man[["box_id", "occlusion"]], on="box_id", how="inner")
            if len(g2) >= 30:
                from sklearn.metrics import cohen_kappa_score
                kappa = cohen_kappa_score(g2["occlusion"], g2["occ_proxy"],
                                          labels=rcfg["occlusion_labels"], weights="linear")
                out["kappa_proxy_vs_manual"] = float(kappa)
                ids_m = g2["image_id"].unique()
                gt_m = gt[gt["image_id"].isin(set(ids_m))].merge(
                    man[["box_id", "occlusion"]], on="box_id", how="left").fillna({"occlusion": "_otro"})
                df = group_eval(gt_m, det, ids_m, classes, "occlusion", rcfg["occlusion_labels"], thr)
                df.insert(0, "split", "trans")
                df.insert(1, "factor", "oclusión (manual)")
                tables.append(df)

    full = pd.concat(tables, ignore_index=True)
    full.to_csv(res / "robustness.csv", index=False)
    cap = ("Robustez por tamaño relativo de la caja y nivel de oclusión "
           f"(recall al umbral operativo {thr:.2f}, IoU 0.5).")
    if "kappa_proxy_vs_manual" in out:
        cap += f" Acuerdo proxy vs. anotación manual: $\\kappa_w$={out['kappa_proxy_vs_manual']:.2f}."
    to_latex(full, tab / "robustness.tex", cap.replace("$\\kappa_w$", "kappa ponderado"), "tab:robustness")
    save_json(out, res / "robustness_summary.json")

    f, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for ax, fac in zip(axes, ["tamaño", "oclusión (proxy)"]):
        d = full[full["factor"] == fac]
        for k, (sp, g) in enumerate(d.groupby("split")):
            x = np.arange(len(g))
            ax.bar(x + (k - 0.5) * 0.4, g["mAP50"], 0.4, label=sp)
            ax.set_xticks(x, g["grupo"])
        ax.set_title(fac)
    axes[0].set_ylabel("mAP@0.5")
    axes[1].legend(frameon=False)
    plt.tight_layout()
    fig.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig / "robustness.pdf"); plt.savefig(fig / "robustness.png", dpi=150)
    plt.close()
    print(full.to_string(index=False))


if __name__ == "__main__":
    main()
