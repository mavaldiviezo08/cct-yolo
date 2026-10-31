"""Genera un mini-dataset sintético en formato COCO Camera Traps (mismo esquema que CCT-20).

Sirve para probar el pipeline completo en minutos y sin GPU (`bash run_all.sh --smoke`).
Cada ubicación tiene un fondo propio (para reproducir el efecto 'memoriza el fondo'),
4 'especies' (formas de colores) con desbalance, ~30 % de fotos vacías, secuencias de
3 fotogramas con fecha/hora, algunas fotos con animal sin caja, y anotaciones en una
resolución distinta a la del archivo (prueba el reescalado de cajas).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data_smoke/raw")
IMG = OUT / "eccv_18_all_images_sm"
ANN = OUT / "eccv_18_annotation_files"
W, H, JSON_SCALE = 320, 240, 2  # el JSON declara 640x480
CATS = [{"id": 0, "name": "empty"}, {"id": 1, "name": "opossum"}, {"id": 2, "name": "raccoon"},
        {"id": 3, "name": "coyote"}, {"id": 4, "name": "bobcat"}]
PROBS = [0.5, 0.3, 0.15, 0.05]  # desbalance
SPLITS = {"train": ([0, 1, 2, 3], 300), "cis_val": ([0, 1, 2, 3], 60), "cis_test": ([0, 1, 2, 3], 90),
          "trans_val": ([10, 11], 45), "trans_test": ([12, 13], 120)}


def background(loc: int, rng) -> np.ndarray:
    g = np.random.default_rng(loc)
    base = g.integers(40, 200, 3)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 3), np.float32)
    for c in range(3):
        img[..., c] = base[c] + 40 * np.sin(xx / (15 + 5 * loc % 7) + c) * np.cos(yy / 25)
    img += rng.normal(0, 6, img.shape)
    for _ in range(4 + loc % 3):  # 'rocas/troncos' fijos por cámara
        cx, cy = g.integers(0, W), g.integers(H // 2, H)
        cv2.ellipse(img, (int(cx), int(cy)), (int(g.integers(15, 40)), int(g.integers(8, 20))), 0, 0, 360,
                    tuple(int(v) for v in g.integers(30, 120, 3)), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def draw(img, cat, rng):
    s = int(rng.integers(14, 60))
    x, y = int(rng.integers(0, W - s)), int(rng.integers(H // 4, H - s))
    col = {1: (60, 60, 230), 2: (230, 80, 60), 3: (60, 200, 60), 4: (40, 220, 230)}[cat]
    if cat == 1:
        cv2.circle(img, (x + s // 2, y + s // 2), s // 2, col, -1)
    elif cat == 2:
        cv2.rectangle(img, (x, y), (x + s, y + s), col, -1)
    elif cat == 3:
        cv2.fillPoly(img, [np.array([[x, y + s], [x + s, y + s], [x + s // 2, y]])], col)
    else:
        cv2.ellipse(img, (x + s // 2, y + s // 2), (s // 2, s // 3), 0, 0, 360, col, -1)
    if rng.random() < 0.3:  # 'vegetación' delante del animal
        for _ in range(6):
            x0 = int(rng.integers(x, x + s))
            cv2.line(img, (x0, y), (x0 + 5, y + s), (20, 90, 20), 2)
    return [x * JSON_SCALE, y * JSON_SCALE, s * JSON_SCALE, s * JSON_SCALE]


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    ANN.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    iid, aid = 0, 0
    for split, (locs, n) in SPLITS.items():
        images, anns = [], []
        t = datetime(2013, 5, 1, 6, 0, 0)
        for k in range(n // 3):
            loc = locs[k % len(locs)]
            t += timedelta(hours=float(rng.uniform(1, 9)))
            empty = rng.random() < 0.3
            cat = int(rng.choice([1, 2, 3, 4], p=PROBS))
            seq = f"seq_{split}_{k}"
            nobox = (not empty) and rng.random() < 0.05
            for fr in range(3):
                img = background(loc, rng)
                name = f"img{iid:06d}.jpg"
                if empty:
                    anns.append(dict(id=f"a{aid}", image_id=str(iid), category_id=0)); aid += 1
                else:
                    bb = draw(img, cat, rng)
                    a = dict(id=f"a{aid}", image_id=str(iid), category_id=cat); aid += 1
                    if not nobox:
                        a["bbox"] = bb
                    anns.append(a)
                cv2.imwrite(str(IMG / name), img)
                images.append(dict(id=str(iid), file_name=name, width=W * JSON_SCALE, height=H * JSON_SCALE,
                                   location=loc, seq_id=seq, frame_num=fr + 1, seq_num_frames=3,
                                   date_captured=(t + timedelta(seconds=fr)).strftime("%Y-%m-%d %H:%M:%S")))
                iid += 1
        with open(ANN / f"{split}_annotations.json", "w") as f:
            json.dump(dict(images=images, annotations=anns, categories=CATS, info={}), f)
    print(f"[sintético] {iid} imágenes en {OUT}")


if __name__ == "__main__":
    main()
