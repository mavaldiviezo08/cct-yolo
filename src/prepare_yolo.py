"""Convierte CCT-20 (formato COCO Camera Traps) a formato YOLO.

Decisiones documentadas en el informe:
* Las imágenes vacías se usan como *imágenes de fondo* (archivo de etiquetas vacío):
  YOLO las trata como negativos puros y aprende a no disparar sobre el fondo.
* Imágenes con animal pero SIN caja (etiqueta solo a nivel de imagen) se excluyen
  del entrenamiento de detección: si se incluyeran, el animal sin caja se
  aprendería como fondo. Se conservan para el análisis a nivel de imagen.
* Las cajas se reescalan al tamaño real del archivo (las imágenes "sm" están
  reducidas a 1024 px de lado máximo).
* Por cada experimento se genera una lista de entrenamiento distinta
  (fracción de fondo, Repeat Factor Sampling para especies raras).

Uso:  python src/prepare_yolo.py
"""
from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from common import has_bbox, load_cct, load_config, p, save_json, seed_everything


def find_file(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"No se encontró {name} bajo {root}. ¿Corriste download_data.py?")
    return hits[0]


def link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:  # Windows sin permisos de symlink
        shutil.copy2(src, dst)


def build_tables(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    raw, img_dir = p(cfg, "raw"), p(cfg, "images")
    if not img_dir.exists():
        cands = [d for d in raw.rglob("*") if d.is_dir() and any(d.glob("*.jpg"))]
        if not cands:
            raise FileNotFoundError(f"No hay imágenes .jpg bajo {raw}")
        img_dir = max(cands, key=lambda d: sum(1 for _ in d.glob("*.jpg")))
    empty_name = cfg["data"]["empty_category"]

    meta_rows, box_rows, all_cats = [], [], set()
    for split, fname in cfg["splits"].items():
        images, anns, cats = load_cct(find_file(raw, fname))
        all_cats |= {c for c in cats.values() if c != empty_name}
        by_img = anns.groupby("image_id")
        for im in images.itertuples(index=False):
            path = img_dir / im.file_name
            if not path.exists():
                path = img_dir / Path(im.file_name).name
            if not path.exists():
                continue
            a = by_img.get_group(im.id) if im.id in by_img.groups else anns.iloc[0:0]
            animal = a[a["category"] != empty_name]
            if animal.empty:
                status = "empty"
            elif all(has_bbox(b) for b in animal["bbox"]):
                status = "animal"
            else:
                status = "animal_nobox"
            # tamaño real del archivo (header JPEG, no decodifica)
            with Image.open(path) as pil:
                W, H = pil.size
            jw = getattr(im, "width", None) or W
            jh = getattr(im, "height", None) or H
            sx, sy = W / jw, H / jh
            species = sorted(set(animal["category"]))
            meta_rows.append(dict(
                image_id=im.id, split=split, path=str(path), width=W, height=H,
                location=str(getattr(im, "location", "")),
                date_captured=getattr(im, "date_captured", None),
                seq_id=getattr(im, "seq_id", None),
                frame_num=getattr(im, "frame_num", None),
                status=status, species="|".join(species), n_boxes=0,
            ))
            if status != "animal":
                continue
            for b in animal.itertuples(index=False):
                x, y, w, h = b.bbox
                x, y, w, h = x * sx, y * sy, w * sx, h * sy
                x, y = max(0.0, x), max(0.0, y)
                w, h = min(w, W - x), min(h, H - y)
                if w <= 1 or h <= 1:
                    continue
                box_rows.append(dict(
                    image_id=im.id, split=split, category=b.category,
                    x=x, y=y, w=w, h=h, rel_area=(w * h) / (W * H),
                    border=int(x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2),
                ))
    meta = pd.DataFrame(meta_rows)
    boxes = pd.DataFrame(box_rows)
    nbox = boxes.groupby(["split", "image_id"]).size().rename("n")
    meta["n_boxes"] = [nbox.get((s, i), 0) for s, i in zip(meta["split"], meta["image_id"])]
    meta.loc[(meta["status"] == "animal") & (meta["n_boxes"] == 0), "status"] = "animal_nobox"
    classes = sorted(all_cats)
    return meta, boxes, classes


def write_yolo_split(meta: pd.DataFrame, boxes: pd.DataFrame, classes: list[str],
                     yolo: Path, split: str) -> None:
    cid = {c: i for i, c in enumerate(classes)}
    (yolo / "images" / split).mkdir(parents=True, exist_ok=True)
    (yolo / "labels" / split).mkdir(parents=True, exist_ok=True)
    bsplit = boxes[boxes["split"] == split].groupby("image_id")
    for r in meta[meta["split"] == split].itertuples(index=False):
        if r.status == "animal_nobox":
            continue
        link(Path(r.path), yolo / "images" / split / f"{r.image_id}.jpg")
        lines = []
        if r.image_id in bsplit.groups:
            for b in bsplit.get_group(r.image_id).itertuples(index=False):
                xc, yc = (b.x + b.w / 2) / r.width, (b.y + b.h / 2) / r.height
                lines.append(f"{cid[b.category]} {xc:.6f} {yc:.6f} {b.w / r.width:.6f} {b.h / r.height:.6f}")
        (yolo / "labels" / split / f"{r.image_id}.txt").write_text("\n".join(lines))


def rfs_factors(meta_tr: pd.DataFrame, t: float) -> pd.Series:
    """Repeat Factor Sampling (Gupta et al., 2019): r_c = max(1, sqrt(t/f_c))."""
    sp = meta_tr["species"].str.split("|")
    n = len(meta_tr)
    f = sp.explode().value_counts() / n
    r_c = np.maximum(1.0, np.sqrt(t / f))
    return sp.apply(lambda s: max(r_c[c] for c in s))


def make_lists(cfg: dict, meta: pd.DataFrame, yolo: Path) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    lists = yolo / "lists"
    lists.mkdir(parents=True, exist_ok=True)
    img = lambda s, i: str(yolo / "images" / s / f"{i}.jpg")

    tr = meta[meta["split"] == "train"]
    tr_animal, tr_empty = tr[tr["status"] == "animal"], tr[tr["status"] == "empty"]
    info = {}
    for exp, ecfg in cfg["experiments"].items():
        files = [img("train", i) for i in tr_animal["image_id"]]
        if ecfg.get("rfs"):
            r = rfs_factors(tr_animal, cfg["data"]["rfs_threshold"])
            extra = 0
            for iid, ri in zip(tr_animal["image_id"], r):
                k = int(math.floor(ri)) - 1 + int(rng.random() < ri - math.floor(ri))
                for rep in range(k):  # copias con nombre propio (evita deduplicación)
                    src_i = yolo / "images" / "train" / f"{iid}.jpg"
                    src_l = yolo / "labels" / "train" / f"{iid}.txt"
                    dst_i = yolo / "images" / "train" / f"{iid}__rfs{rep}.jpg"
                    if not dst_i.exists():
                        link(src_i, dst_i)
                        shutil.copy2(src_l, yolo / "labels" / "train" / f"{iid}__rfs{rep}.txt")
                    files.append(str(dst_i))
                    extra += 1
            info[f"{exp}_rfs_extra"] = extra
        n_bg = int(round(ecfg["background_fraction"] * len(tr_animal)))
        n_bg = min(n_bg, len(tr_empty))
        bg = rng.choice(tr_empty["image_id"].to_numpy(), size=n_bg, replace=False) if n_bg else []
        files += [img("train", i) for i in bg]
        (lists / f"train_{exp}.txt").write_text("\n".join(files))
        info[exp] = dict(n_train=len(files), n_background=n_bg)

    def eval_list(split, cap):
        m = meta[(meta["split"] == split) & (meta["status"] != "animal_nobox")]
        if cap and len(m) > cap:
            m = m.sample(cap, random_state=cfg["seed"])
        (lists / f"{split}.txt").write_text("\n".join(img(split, i) for i in m["image_id"]))
        return len(m)

    info["cis_val"] = eval_list("cis_val", cfg["data"]["max_val_images"])
    for s in ["cis_test", "trans_test", "trans_val"]:
        info[s] = eval_list(s, cfg["data"]["max_test_images"])
    return info


def write_yamls(cfg: dict, yolo: Path, classes: list[str]) -> None:
    names = {i: c for i, c in enumerate(classes)}
    for exp in cfg["experiments"]:
        d = dict(path=str(yolo), train=str(yolo / "lists" / f"train_{exp}.txt"),
                 val=str(yolo / "lists" / "cis_val.txt"), names=names)
        for s in ["cis_test", "trans_test"]:
            d[s] = str(yolo / "lists" / f"{s}.txt")
        (yolo / f"data_{exp}.yaml").write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False))
    # YAMLs de evaluación: 'val' apunta al split a evaluar
    for s in ["cis_val", "cis_test", "trans_test"]:
        d = dict(path=str(yolo), train=str(yolo / "lists" / "cis_val.txt"),
                 val=str(yolo / "lists" / f"{s}.txt"), names=names)
        (yolo / f"eval_{s}.yaml").write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False))


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    yolo, res = p(cfg, "yolo"), p(cfg, "results")
    res.mkdir(parents=True, exist_ok=True)

    meta, boxes, classes = build_tables(cfg)
    meta.to_csv(res / "metadata.csv", index=False)
    boxes.to_csv(res / "boxes.csv", index=False)
    save_json(classes, yolo / "classes.json")
    print(f"[clases] {len(classes)}: {classes}")
    for split in cfg["splits"]:
        write_yolo_split(meta, boxes, classes, yolo, split)
    info = make_lists(cfg, meta, yolo)
    write_yamls(cfg, yolo, classes)
    summary = meta.groupby(["split", "status"]).size().unstack(fill_value=0)
    print(summary)
    info["status_por_split"] = summary.to_dict()
    save_json(info, res / "prepare_summary.json")
    print("[listo] dataset YOLO en", yolo)


if __name__ == "__main__":
    main()
