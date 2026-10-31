"""Inferencia por lotes con caché en CSV (una fila por detección)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import load_config, p


def gt_frame(boxes: pd.DataFrame, split: str) -> pd.DataFrame:
    g = boxes[boxes["split"] == split].copy()
    g["x1"], g["y1"] = g["x"], g["y"]
    g["x2"], g["y2"] = g["x"] + g["w"], g["y"] + g["h"]
    return g


def read_list(list_file: Path) -> list[str]:
    return [l.strip() for l in Path(list_file).read_text().splitlines() if l.strip()]


def predict_split(weights: Path, split: str, exp: str, cfg: dict | None = None,
                  force: bool = False, batch: int = 32) -> pd.DataFrame:
    cfg = cfg or load_config()
    out = p(cfg, "results") / "preds" / f"{exp}_{split}.csv"
    if out.exists() and not force:
        return pd.read_csv(out, dtype={"image_id": str})
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = model.names
    files = read_list(p(cfg, "yolo") / "lists" / f"{split}.txt")
    rows = []
    e = cfg["eval"]
    for i in range(0, len(files), batch):
        for r in model.predict(files[i:i + batch], conf=e["conf_threshold"], iou=e["iou_nms"],
                               imgsz=cfg["train"]["imgsz"], max_det=100, verbose=False):
            iid = Path(r.path).stem
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            for (x1, y1, x2, y2), c, k in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist()):
                rows.append((iid, names[int(k)], x1, y1, x2, y2, c))
        print(f"\r  [{exp}/{split}] {min(i + batch, len(files))}/{len(files)}", end="", flush=True)
    print()
    df = pd.DataFrame(rows, columns=["image_id", "category", "x1", "y1", "x2", "y2", "score"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df
