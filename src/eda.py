"""Análisis exploratorio: desbalance de especies, imágenes vacías, tamaños de caja.

Genera tablas LaTeX (report/tables) y figuras (report/figures).
Uso:  python src/eda.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import load_config, p, save_json, to_latex

SPLITS = ["train", "cis_val", "cis_test", "trans_val", "trans_test"]


def main() -> None:
    cfg = load_config()
    res, tab, fig = p(cfg, "results"), p(cfg, "report_tables"), p(cfg, "report_figures")
    fig.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(res / "metadata.csv", dtype={"image_id": str})
    boxes = pd.read_csv(res / "boxes.csv", dtype={"image_id": str})

    # 1) Resumen por split
    g = meta.groupby("split")
    summ = pd.DataFrame({
        "imágenes": g.size(),
        "ubicaciones": g["location"].nunique(),
        "% vacías": g["status"].apply(lambda s: 100 * (s == "empty").mean()),
        "con caja": g["status"].apply(lambda s: (s == "animal").sum()),
        "animal sin caja": g["status"].apply(lambda s: (s == "animal_nobox").sum()),
    }).reindex([s for s in SPLITS if s in g.groups])
    summ["cajas"] = boxes.groupby("split").size().reindex(summ.index).fillna(0).astype(int)
    to_latex(summ, tab / "eda_splits.tex", "Composición de las particiones de CCT-20.",
             "tab:splits", float_fmt="{:.1f}", index=True)

    # 2) Prevalencia de especies (cajas) por split
    prev = boxes.pivot_table(index="category", columns="split", values="x",
                             aggfunc="count", fill_value=0)
    prev = prev.reindex(columns=[s for s in SPLITS if s in prev.columns])
    prev = prev.sort_values("train", ascending=False)
    prev["% train"] = 100 * prev["train"] / prev["train"].sum()
    imb = float(prev["train"].max() / max(1, prev["train"][prev["train"] > 0].min()))
    to_latex(prev, tab / "eda_species.tex",
             f"Número de cajas por especie y partición (razón de desbalance en train: {imb:.0f}:1).",
             "tab:species", float_fmt="{:.1f}", index=True)

    fig_, ax = plt.subplots(figsize=(7, 3.6))
    prev["train"].plot.bar(ax=ax, color="#3b6e8f", log=True)
    ax.set_ylabel("cajas en train (escala log)")
    ax.set_xlabel("")
    ax.set_title("Distribución de especies en entrenamiento")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(fig / "species_distribution.pdf")
    plt.savefig(fig / "species_distribution.png", dpi=150)
    plt.close()

    # 3) Tamaño relativo de las cajas
    fig_, ax = plt.subplots(figsize=(6, 3))
    for s in ["train", "cis_test", "trans_test"]:
        b = boxes[boxes["split"] == s]["rel_area"]
        if len(b):
            ax.hist(b.clip(1e-4, 1), bins=50, histtype="step", density=True, label=s, log=False)
    ax.set_xscale("log")
    ax.set_xlabel("área de la caja / área de la imagen")
    ax.set_ylabel("densidad")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig / "box_sizes.pdf")
    plt.savefig(fig / "box_sizes.png", dpi=150)
    plt.close()

    # 4) Imágenes por ubicación (memorizar el fondo es más fácil si pocas cámaras dominan)
    loc = meta[meta["split"] == "train"].groupby("location").size().sort_values(ascending=False)
    save_json({"imbalance_ratio_train": imb,
               "train_images_per_location": loc.to_dict(),
               "top3_location_share": float(loc.head(3).sum() / loc.sum()) if len(loc) else None},
              res / "eda_summary.json")
    print(summ)
    print(prev)


if __name__ == "__main__":
    main()
