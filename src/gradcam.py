"""¿El modelo mira al animal o al fondo? (entregable 3, evidencia)

1. Grad-CAM sobre una capa del backbone de YOLOv8 (implementación propia con hooks):
   el objetivo es el score de clase de la ancla con mayor activación para la especie.
   Métrica cuantitativa: *energía dentro de la caja* = fracción de la masa del mapa
   que cae dentro de la caja GT, y su razón con el área relativa de la caja
   (1.0 = no mejor que un mapa uniforme). Se compara cis vs. trans.
2. Test de enmascarado del animal: se borra el animal (inpainting de OpenCV sobre la
   caja GT) y se vuelve a detectar. Si el modelo sigue "viendo" la misma especie con
   confianza >= umbral, la decisión se apoya en el fondo de la escena.
3. Galería: aciertos cis, aciertos trans, errores trans y falsos positivos en fotos
   vacías de cámaras nuevas.

Uso:  python src/gradcam.py [--exp baseline]
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
import torch

from common import load_config, p, save_json, seed_everything, to_latex
from predict import predict_split, read_list


class GradCAM:
    def __init__(self, weights: Path, layer: int, imgsz: int = 640):
        from ultralytics import YOLO

        yolo = YOLO(str(weights))
        self.names = yolo.names
        self.cls_id = {v: k for k, v in self.names.items()}
        self.model = yolo.model.float().eval()
        for prm in self.model.parameters():
            prm.requires_grad_(True)
        self.imgsz = imgsz
        self.act = None
        self.model.model[layer].register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        out.retain_grad()
        self.act = out

    def letterbox(self, img):
        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
        return canvas, (top, left, nh, nw)

    def __call__(self, img_bgr: np.ndarray, category: str) -> tuple[np.ndarray, float]:
        lb, (top, left, nh, nw) = self.letterbox(img_bgr)
        x = torch.from_numpy(lb[:, :, ::-1].copy()).permute(2, 0, 1)[None].float() / 255
        self.model.zero_grad()
        out = self.model(x)
        y = out[0] if isinstance(out, (list, tuple)) else out  # (1, 4+nc, A)
        c = self.cls_id[category]
        scores = y[0, 4 + c]
        a = int(torch.argmax(scores))
        scores[a].backward()
        A, G = self.act[0], self.act.grad[0]
        cam = torch.relu((G.mean((1, 2))[:, None, None] * A).sum(0)).detach().numpy()
        cam = cv2.resize(cam, (self.imgsz, self.imgsz))[top:top + nh, left:left + nw]
        cam = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
        cam = cam / (cam.max() + 1e-9)
        return cam, float(scores[a].detach())


def energy_in_box(cam, b):
    x, y, w, h = (int(round(v)) for v in b)
    tot = cam.sum() + 1e-9
    return float(cam[max(0, y):y + h, max(0, x):x + w].sum() / tot)


def overlay(img, cam, box=None, title=""):
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out = cv2.addWeighted(img, 0.55, heat, 0.45, 0)
    if box is not None:
        x, y, w, h = (int(v) for v in box)
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 255), 3)
    return out[:, :, ::-1], title


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="baseline")
    args = ap.parse_args()
    cfg = load_config()
    seed_everything(cfg["seed"])
    gc = cfg["gradcam"]
    res, tab, fig = p(cfg, "results"), p(cfg, "report_tables"), p(cfg, "report_figures")
    fig.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(res / "metadata.csv", dtype={"image_id": str})
    boxes = pd.read_csv(res / "boxes.csv", dtype={"image_id": str})
    thr = json.loads((res / "operating_threshold.json").read_text())["threshold"] \
        if (res / "operating_threshold.json").exists() else cfg["eval"]["operating_conf"]
    w = p(cfg, "runs") / args.exp / "weights" / "best.pt"
    cam_model = GradCAM(w, gc["layer"], cfg["train"]["imgsz"])
    from ultralytics import YOLO
    det_model = YOLO(str(w))
    rng = np.random.default_rng(cfg["seed"])

    rows, gallery, mask_rows = [], {}, []
    for split in ["cis_test", "trans_test"]:
        ids = {Path(f).stem for f in read_list(p(cfg, "yolo") / "lists" / f"{split}.txt")}
        det = predict_split(w, split, args.exp, cfg)
        top = det.sort_values("score", ascending=False).drop_duplicates("image_id").set_index("image_id")
        b = boxes[(boxes["split"] == split) & boxes["image_id"].isin(ids)]
        # caja más grande por imagen como objetivo
        b = b.sort_values("rel_area", ascending=False).drop_duplicates("image_id")
        sample = b.sample(min(len(b), gc["n_mask_test"]), random_state=cfg["seed"])
        paths = meta.set_index("image_id")["path"]
        for r in sample.itertuples():
            img = cv2.imread(paths[r.image_id])
            if img is None:
                continue
            cam, s = cam_model(img, r.category)
            e = energy_in_box(cam, (r.x, r.y, r.w, r.h))
            pred = top.loc[r.image_id] if r.image_id in top.index else None
            correct = pred is not None and pred["category"] == r.category and pred["score"] >= thr
            rows.append(dict(split=split, image_id=r.image_id, category=r.category,
                             energy_in_box=e, rel_area=r.rel_area,
                             concentration=e / max(r.rel_area, 1e-6), correct=correct))
            # --- test de enmascarado (inpainting del animal) ---
            mask = np.zeros(img.shape[:2], np.uint8)
            x0, y0 = int(max(0, r.x)), int(max(0, r.y))
            mask[y0:int(r.y + r.h), x0:int(r.x + r.w)] = 255
            masked = cv2.inpaint(img, cv2.dilate(mask, np.ones((9, 9), np.uint8)), 5, cv2.INPAINT_TELEA)
            pr = det_model.predict(masked, conf=0.05, verbose=False,  # BGR, como cv2
                                   imgsz=cfg["train"]["imgsz"])[0]
            same = [float(c) for c, k in zip(pr.boxes.conf.tolist(), pr.boxes.cls.tolist())
                    if det_model.names[int(k)] == r.category]
            mask_rows.append(dict(split=split, image_id=r.image_id, category=r.category,
                                  orig_score=float(pred["score"]) if pred is not None else 0.0,
                                  masked_score=max(same, default=0.0)))
            key = (split, "acierto" if correct else "error")
            if len(gallery.setdefault(key, [])) < gc["n_examples"] // 4:
                gallery[key].append(overlay(img, cam, (r.x, r.y, r.w, r.h),
                                            f"{split.split('_')[0]} | {r.category} | E={e:.2f}"))
        # falsos positivos con alta confianza en fotos vacías de cámaras nuevas
        if split == "trans_test":
            emp = meta[(meta["split"] == split) & (meta["status"] == "empty") & meta["image_id"].isin(ids)]
            fp = top[top.index.isin(emp["image_id"]) & (top["score"] >= thr)].sort_values("score", ascending=False)
            for iid, fr in fp.head(gc["n_examples"] // 4).iterrows():
                img = cv2.imread(paths[iid])
                cam, _ = cam_model(img, fr["category"])
                gallery.setdefault(("trans_test", "FP en foto vacía"), []).append(
                    overlay(img, cam, None, f"vacía -> {fr['category']} ({fr['score']:.2f})"))

    df = pd.DataFrame(rows)
    df.to_csv(res / "gradcam_energy.csv", index=False)
    mk = pd.DataFrame(mask_rows)
    mk["persiste"] = mk["masked_score"] >= thr
    mk.to_csv(res / "mask_test.csv", index=False)

    summ = df.groupby(["split", "correct"]).agg(
        n=("image_id", "size"), energia_en_caja=("energy_in_box", "mean"),
        area_rel=("rel_area", "mean"), concentracion=("concentration", "median")).reset_index()
    msum = mk.groupby("split").agg(n=("image_id", "size"),
                                   score_original=("orig_score", "mean"),
                                   score_enmascarado=("masked_score", "mean"),
                                   pct_persiste=("persiste", "mean")).reset_index()
    msum["pct_persiste"] *= 100
    to_latex(summ, tab / "gradcam.tex",
             "Grad-CAM: fracción de la energía del mapa dentro de la caja GT (concentración = energía / área relativa; 1 = uniforme).",
             "tab:gradcam")
    to_latex(msum, tab / "mask_test.tex",
             f"Test de enmascarado: el animal se borra por inpainting; \\% persiste = la misma especie sigue detectada con score $\\geq$ {thr:.2f}.",
             "tab:masktest", float_fmt="{:.2f}")
    save_json({"energy": summ.to_dict("records"), "mask_test": msum.to_dict("records")}, res / "gradcam_summary.json")

    # galería
    items = [(k, v) for k, lst in gallery.items() for v in lst]
    if items:
        n = len(items)
        cols = 4
        rws = int(np.ceil(n / cols))
        f, axes = plt.subplots(rws, cols, figsize=(3.2 * cols, 2.5 * rws))
        for ax in np.atleast_1d(axes).ravel():
            ax.axis("off")
        for ax, ((split, kind), (im, ttl)) in zip(np.atleast_1d(axes).ravel(), items):
            ax.imshow(im)
            ax.set_title(f"{kind}\n{ttl}", fontsize=7)
        plt.tight_layout()
        plt.savefig(fig / "gradcam_gallery.pdf", dpi=150)
        plt.savefig(fig / "gradcam_gallery.png", dpi=110)
        plt.close()
    print(summ.to_string(index=False))
    print(msum.to_string(index=False))


if __name__ == "__main__":
    main()
