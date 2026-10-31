"""Descarga el benchmark CCT-20 desde LILA BC (~6 GB de imágenes + 3 MB de anotaciones).

Uso:  python src/download_data.py [--skip-images]
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

from common import load_config, p


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[ok] ya existe {dest.name}")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[descargando] {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e9:.2f}/{total / 1e9:.2f} GB", end="", flush=True)
    print()
    tmp.rename(dest)
    return dest


def safe_extract(tar_path: Path, out: Path) -> None:
    """Extrae evitando rutas que escapen del directorio destino."""
    out = out.resolve()
    with tarfile.open(tar_path) as tar:
        for m in tar.getmembers():
            target = (out / m.name).resolve()
            if not str(target).startswith(str(out)):
                raise RuntimeError(f"Ruta sospechosa en el tar: {m.name}")
        tar.extractall(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-images", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    raw = p(cfg, "raw")
    raw.mkdir(parents=True, exist_ok=True)

    ann_tar = download(cfg["download"]["annotations_url"], raw / "eccv_18_annotations.tar.gz")
    safe_extract(ann_tar, raw)
    if not args.skip_images:
        img_tar = download(cfg["download"]["images_url"], raw / "eccv_18_all_images_sm.tar.gz")
        if not p(cfg, "images").exists():
            print("[extrayendo] imágenes (puede tardar ~10 min)")
            safe_extract(img_tar, raw)
    free = shutil.disk_usage(raw).free / 1e9
    print(f"[listo] datos en {raw}  (espacio libre: {free:.1f} GB)")


if __name__ == "__main__":
    main()
