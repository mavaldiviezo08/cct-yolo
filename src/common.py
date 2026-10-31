"""Utilidades compartidas: configuración, lectura de anotaciones CCT, tablas LaTeX."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path or os.environ.get("CCT_CONFIG") or ROOT / "configs" / "config.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Permite redirigir rutas (p.ej. smoke test) con variables de entorno
    for key in list(cfg["paths"]):
        env = os.environ.get(f"CCT_{key.upper()}")
        if env:
            cfg["paths"][key] = env
    cfg["_root"] = ROOT
    return cfg


def p(cfg: dict, key: str) -> Path:
    """Ruta absoluta para cfg['paths'][key]."""
    path = Path(cfg["paths"][key])
    return path if path.is_absolute() else ROOT / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import sys

    if "torch" in sys.modules:  # solo si el script ya usa torch (evita cargarlo en vano)
        sys.modules["torch"].manual_seed(seed)


def load_cct(json_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Lee un JSON en formato COCO Camera Traps.

    Devuelve (imágenes, anotaciones, {category_id: nombre}).
    """
    with open(json_path, encoding="utf-8") as f:
        d = json.load(f)
    images = pd.DataFrame(d["images"])
    anns = pd.DataFrame(d.get("annotations", []))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    if anns.empty:
        anns = pd.DataFrame(columns=["id", "image_id", "category_id", "bbox"])
    if "bbox" not in anns.columns:
        anns["bbox"] = None
    images["id"] = images["id"].astype(str)
    anns["image_id"] = anns["image_id"].astype(str)
    anns["category"] = anns["category_id"].map(cats)
    return images, anns, cats


def has_bbox(b) -> bool:
    return isinstance(b, (list, tuple)) and len(b) == 4 and b[2] > 0 and b[3] > 0


def to_latex(df: pd.DataFrame, path: Path, caption: str, label: str,
             float_fmt: str = "{:.3f}", index: bool = False) -> None:
    """Escribe un DataFrame como tabla booktabs lista para \\input en el informe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = df.to_latex(index=index, escape=True,
                       float_format=lambda x: float_fmt.format(x),
                       na_rep="--")
    body = body.strip()
    wide = df.shape[1] + int(index) > 6  # solo se escala si la tabla es ancha
    inner = f"\\resizebox{{\\linewidth}}{{!}}{{%\n{body}\n}}" if wide else body
    tex = (
        "\\begin{table}[!htbp]\n\\centering\n\\small\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n{inner}\n\\end{{table}}\n"
    )
    path.write_text(tex, encoding="utf-8")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)
