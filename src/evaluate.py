"""Evaluación cuantitativa (entregables 2 y 3).

* mAP@0.5 y mAP@0.5:0.95 por especie en cis_test (ubicaciones vistas) y
  trans_test (ubicaciones nuevas) -> brecha de generalización.
* Efecto del desbalance: AP vs. número de cajas de entrenamiento por especie.
* Tabla de ablación entre experimentos (fondo, RFS).
* Umbral operativo elegido en cis_val maximizando F1 a nivel de imagen
  ("¿hay fauna en la foto?"), reportando precisión/recall/F1 y tasa de falsos
  positivos en fotos vacías.

Uso:  python src/evaluate.py [--main baseline]
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_config, p, save_json, to_latex
from metrics import best_threshold, evaluate, image_level, summarize
from predict import gt_frame, predict_split, read_list


def split_meta(meta, cfg, split):
    ids = {Path(f).stem for f in read_list(p(cfg, "yolo") / "lists" / f"{split}.txt")}
    return meta[(meta["split"] == split) & meta["image_id"].isin(ids)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="baseline")
    args = ap.parse_args()
    cfg = load_config()
    res, tab, fig = p(cfg, "results"), p(cfg, "report_tables"), p(cfg, "report_figures")
    fig.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(res / "metadata.csv", dtype={"image_id": str})
    boxes = pd.read_csv(res / "boxes.csv", dtype={"image_id": str})
    classes = json.loads((p(cfg, "yolo") / "classes.json").read_text())
    train_counts = boxes[boxes["split"] == "train"]["category"].value_counts()
    rare = set(train_counts[train_counts <= train_counts.quantile(1 / 3)].index)

    ablation, all_metrics = [], {}
    for exp in cfg["experiments"]:
        w = p(cfg, "runs") / exp / "weights" / "best.pt"
        if not w.exists():
            print(f"[omitido] {exp}: no hay {w}")
            continue
        m_val = split_meta(meta, cfg, "cis_val")
        thr = best_threshold(m_val, predict_split(w, "cis_val", exp, cfg))
        out = {"operating_threshold": thr}
        per = {}
        for split in ["cis_test", "trans_test"]:
            ms = split_meta(meta, cfg, split)
            det = predict_split(w, split, exp, cfg)
            pc = evaluate(gt_frame(boxes, split), det, ms["image_id"], classes)
            per[split] = pc
            s = summarize(pc)
            s["mAP50_rare"] = float(pc[pc["category"].isin(rare) & (pc["n_gt"] > 0)]["AP50"].mean())
            s["image_level"] = image_level(ms, det, thr)
            # mAP50 por ubicación (dispersión entre cámaras)
            loc = {}
            for l, g in ms.groupby("location"):
                if (g["status"] == "animal").sum() >= 20:
                    loc[str(l)] = summarize(evaluate(gt_frame(boxes, split), det, g["image_id"], classes))["mAP50"]
            s["mAP50_by_location"] = loc
            out[split] = s
            pc.to_csv(res / f"per_class_{exp}_{split}.csv", index=False)
        all_metrics[exp] = out
        c, t = out["cis_test"], out["trans_test"]
        ablation.append({
            "experimento": exp,
            "mAP50 cis": c["mAP50"], "mAP50:95 cis": c["mAP50_95"],
            "mAP50 trans": t["mAP50"], "mAP50:95 trans": t["mAP50_95"],
            "brecha mAP50": c["mAP50"] - t["mAP50"],
            "mAP50 raras (trans)": t["mAP50_rare"],
            "FP vacías trans": t["image_level"]["empty_fp_rate"],
        })
        if exp == args.main:
            save_json({"exp": exp, "threshold": thr}, res / "operating_threshold.json")
            # Tabla por especie
            sp = pd.DataFrame({"cajas train": train_counts.reindex(classes).fillna(0).astype(int)})
            for split, tag in [("cis_test", "cis"), ("trans_test", "trans")]:
                pc = per[split].set_index("category")
                sp[f"AP50 {tag}"] = pc["AP50"]
                sp[f"AP50:95 {tag}"] = pc["AP50_95"]
            sp = sp.sort_values("cajas train", ascending=False)
            sp.index = [f"{c} *" if c in rare else c for c in sp.index]
            sp.to_csv(res / "per_species_main.csv")
            to_latex(sp, tab / "per_species.tex",
                     f"AP por especie del modelo {exp} en ubicaciones vistas (cis) y nuevas (trans). "
                     "(*): tercio de especies con menos cajas de entrenamiento.",
                     "tab:perspecies", index=True)
            # Imagen: métricas a nivel de imagen
            il = pd.DataFrame({s: out[s]["image_level"] for s in ["cis_test", "trans_test"]}).T
            il = il[["threshold", "precision", "recall", "f1", "empty_fp_rate", "auc"]]
            to_latex(il, tab / "image_level.tex",
                     f"Filtrado de fotos vacías a nivel de imagen con umbral {thr:.3f} (elegido en cis\\_val maximizando F1).",
                     "tab:imagelevel", index=True)
            _plots(sp, train_counts, per, fig, rare, out)

    if ablation:
        df = pd.DataFrame(ablation)
        df.to_csv(res / "ablation.csv", index=False)
        to_latex(df, tab / "ablation.tex",
                 "Ablación: fracción de imágenes de fondo y Repeat Factor Sampling (RFS).",
                 "tab:ablation")
        print(df.to_string(index=False))
    save_json(all_metrics, res / "metrics.json")


def _plots(sp, train_counts, per, fig, rare, out):
    # 1) AP50 por especie cis vs trans
    cats = list(train_counts.index)
    cis = per["cis_test"].set_index("category").reindex(cats)["AP50"]
    tra = per["trans_test"].set_index("category").reindex(cats)["AP50"]
    x = np.arange(len(cats))
    f, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(x - 0.2, cis, 0.4, label="cis (ubicaciones vistas)", color="#3b6e8f")
    ax.bar(x + 0.2, tra, 0.4, label="trans (ubicaciones nuevas)", color="#d08c3c")
    ax.set_xticks(x, [c + ("*" if c in rare else "") for c in cats], rotation=60, ha="right")
    ax.set_ylabel("AP@0.5")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.set_title("AP por especie (ordenadas por frecuencia en train; * = raras)")
    plt.tight_layout()
    plt.savefig(fig / "ap_per_species.pdf"); plt.savefig(fig / "ap_per_species.png", dpi=150)
    plt.close()
    # 2) AP vs nº de cajas de entrenamiento
    f, ax = plt.subplots(figsize=(5, 3.4))
    n = train_counts.reindex(cats)
    ax.scatter(n, cis, label="cis", color="#3b6e8f")
    ax.scatter(n, tra, label="trans", color="#d08c3c", marker="s")
    for c in cats:
        ax.annotate(c, (n[c], tra[c]), fontsize=6, alpha=.7)
    ax.set_xscale("log"); ax.set_xlabel("cajas de entrenamiento (log)"); ax.set_ylabel("AP@0.5")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig / "ap_vs_frequency.pdf"); plt.savefig(fig / "ap_vs_frequency.png", dpi=150)
    plt.close()
    # 3) mAP50 por ubicación
    loc = {k: v for k, v in out["trans_test"]["mAP50_by_location"].items()}
    locc = {k: v for k, v in out["cis_test"]["mAP50_by_location"].items()}
    if loc or locc:
        f, ax = plt.subplots(figsize=(5, 3))
        ax.boxplot([list(locc.values()) or [np.nan], list(loc.values()) or [np.nan]])
        ax.set_xticks([1, 2], ["cis", "trans"])
        ax.set_ylabel("mAP@0.5 por cámara")
        plt.tight_layout()
        plt.savefig(fig / "map_by_location.pdf"); plt.savefig(fig / "map_by_location.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
