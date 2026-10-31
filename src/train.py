"""Entrena YOLOv8 sobre CCT-20 para un experimento de ablación.

Uso:  python src/train.py --exp baseline   (ver experiments en configs/config.yaml)
      python src/train.py --exp rfs --epochs 20 --model yolov8n.pt
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO

from common import load_config, p, save_json, seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="baseline")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--model")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--device")
    args = ap.parse_args()

    cfg = load_config()
    seed_everything(cfg["seed"])
    t = cfg["train"]
    epochs = args.epochs or t["epochs"]
    model_name = args.model or t["model"]
    data = p(cfg, "yolo") / f"data_{args.exp}.yaml"

    model = YOLO(model_name)
    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=args.imgsz or t["imgsz"],
        batch=args.batch or t["batch"],
        patience=t["patience"],
        workers=t["workers"],
        device=args.device or t["device"],
        project=str(p(cfg, "runs")),
        name=args.exp,
        exist_ok=True,
        seed=cfg["seed"],
        deterministic=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        plots=True,
        verbose=True,
        **t["augment"],
    )
    save_json({"exp": args.exp, "model": model_name, "epochs": epochs,
               "experiment_cfg": cfg["experiments"][args.exp], "augment": t["augment"]},
              p(cfg, "runs") / args.exp / "train_config.json")


if __name__ == "__main__":
    main()
