"""Módulo de monitoreo ecológico basado en reglas (entregable 5).

Toma TODAS las fotos de una cámara no vista en entrenamiento (ubicación trans),
corre el detector y agrega las detecciones como lo haría un equipo de manejo de
fauna. Reglas:

R1 Filtro de confianza por especie (umbral operativo de cis_val por defecto;
   se puede bajar para especies prioritarias en configs -> monitor.conf_per_class).
R2 Supresión entre clases: dos cajas de especies distintas con IoU > 0.7 son el
   mismo animal -> se conserva la de mayor confianza.
R3 Foto "con fauna" si sobrevive al menos una detección; si no, "vacía".
R4 Secuencia (disparo) = mismo seq_id; sin seq_id, fotos con < seq_gap_seconds.
   Especie de la secuencia = mayor suma de confianzas; individuos = MaxN
   (máximo número simultáneo en un fotograma, estándar en ecología de cámaras trampa).
R5 Eventos independientes: secuencias de la misma especie en la misma cámara
   separadas por menos de `independence_minutes` (30 min) se fusionan.
R6 Indicadores: % fotos vacías, riqueza de especies, eventos independientes,
   Índice de Abundancia Relativa RAI = eventos / días-trampa × 100, y patrón
   de actividad diurno/nocturno por especie.

Como las cámaras de CCT-20 tienen etiquetas, se aplica exactamente el mismo motor
de reglas a la verdad de campo y se comparan los dos reportes: así se mide cómo
los errores del detector se propagan a los indicadores ecológicos.

Uso:  python src/monitor.py [--exp baseline] [--location 46]
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_config, p, save_json, to_latex
from metrics import iou_matrix

# ----------------------------------------------------------------- motor de reglas


def frame_table(meta: pd.DataFrame, det: pd.DataFrame, mcfg: dict, base_thr: float) -> pd.DataFrame:
    """R1-R3: de detecciones crudas a una fila por foto con conteo por especie."""
    thr = det["category"].map(mcfg.get("conf_per_class") or {}).fillna(base_thr)
    d = det[det["score"] >= thr].copy()
    keep = []
    cols = ["x1", "y1", "x2", "y2"]
    for _, g in d.groupby("image_id"):
        g = g.sort_values("score", ascending=False)
        ious = iou_matrix(g[cols].to_numpy(float), g[cols].to_numpy(float))
        alive = np.ones(len(g), bool)
        for i in range(len(g)):
            if alive[i]:
                alive[i + 1:] &= ~(ious[i, i + 1:] > 0.7)
        keep.append(g[alive])
    d = pd.concat(keep) if keep else d.iloc[0:0]
    cnt = d.groupby(["image_id", "category"]).agg(n=("score", "size"), conf=("score", "sum")).reset_index()
    f = meta[["image_id", "location", "datetime", "seq_id"]].copy()
    f["con_fauna"] = f["image_id"].isin(cnt["image_id"])
    return f, cnt


def sequences(f: pd.DataFrame, cnt: pd.DataFrame, gap_s: int) -> pd.DataFrame:
    """R4: agrupa fotos en secuencias y asigna especie + MaxN."""
    f = f.sort_values("datetime").copy()
    if f["seq_id"].notna().all():
        f["seq"] = f["seq_id"].astype(str)
    else:
        new = f["datetime"].diff().dt.total_seconds().fillna(1e9) > gap_s
        f["seq"] = "auto" + new.cumsum().astype(str)
    c = cnt.merge(f[["image_id", "seq", "datetime", "location"]], on="image_id")
    if c.empty:
        return pd.DataFrame(columns=["seq", "location", "category", "maxn", "start"])
    agg = c.groupby(["seq", "category"]).agg(conf=("conf", "sum"), maxn=("n", "max")).reset_index()
    best = agg.sort_values("conf", ascending=False).drop_duplicates("seq")
    t0 = f.groupby("seq").agg(start=("datetime", "min"), location=("location", "first"))
    return best.merge(t0, left_on="seq", right_index=True)[["seq", "location", "category", "maxn", "start"]]


def independent_events(seqs: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """R5: fusiona secuencias de la misma especie separadas < `minutes`."""
    out = []
    for (_, _), g in seqs.sort_values("start").groupby(["location", "category"]):
        last, cur = None, None
        for r in g.itertuples(index=False):
            if last is not None and pd.notna(r.start) and pd.notna(last) and \
                    (r.start - last).total_seconds() < minutes * 60:
                cur["maxn"] = max(cur["maxn"], r.maxn)
            else:
                if cur:
                    out.append(cur)
                cur = dict(location=r.location, category=r.category, start=r.start, maxn=r.maxn)
            last = r.start
        if cur:
            out.append(cur)
    return pd.DataFrame(out, columns=["location", "category", "start", "maxn"])


def build_report(f, cnt, mcfg) -> dict:
    seqs = sequences(f, cnt, mcfg["seq_gap_seconds"])
    ev = independent_events(seqs, mcfg["independence_minutes"])
    trap_days = max(1, f["datetime"].dt.normalize().nunique())
    n0, n1 = mcfg["night_hours"]
    if not ev.empty:
        h = ev["start"].dt.hour
        ev["nocturno"] = (h >= n0) | (h < n1)
        ev["periodo"] = ev["start"].dt.to_period(mcfg["period"]).astype(str)
    cam = dict(
        fotos=len(f), pct_vacias=round(100 * (1 - f["con_fauna"].mean()), 1),
        secuencias_con_fauna=int(seqs["seq"].nunique()) if len(seqs) else 0,
        eventos_indep=len(ev), riqueza=int(ev["category"].nunique()) if len(ev) else 0,
        dias_trampa=trap_days,
        desde=str(f["datetime"].min().date()) if f["datetime"].notna().any() else "",
        hasta=str(f["datetime"].max().date()) if f["datetime"].notna().any() else "",
    )
    if ev.empty:
        sp = pd.DataFrame(columns=["eventos", "individuos_MaxN", "RAI", "pct_nocturno"])
        per = pd.DataFrame()
    else:
        sp = ev.groupby("category").agg(eventos=("start", "size"), individuos_MaxN=("maxn", "sum"),
                                        pct_nocturno=("nocturno", "mean"))
        sp["RAI"] = 100 * sp["eventos"] / trap_days
        sp["pct_nocturno"] *= 100
        sp = sp[["eventos", "individuos_MaxN", "RAI", "pct_nocturno"]].sort_values("eventos", ascending=False)
        per = ev.pivot_table(index="periodo", columns="category", values="maxn",
                             aggfunc="size", fill_value=0)
    return dict(camera=cam, species=sp, periods=per, events=ev)


# ---------------------------------------------------------------------- script


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="baseline")
    ap.add_argument("--location", default=None)
    args = ap.parse_args()
    cfg = load_config()
    mcfg = cfg["monitor"]
    res, tab, fig = p(cfg, "results"), p(cfg, "report_tables"), p(cfg, "report_figures")
    meta = pd.read_csv(res / "metadata.csv", dtype={"image_id": str, "location": str})
    boxes = pd.read_csv(res / "boxes.csv", dtype={"image_id": str})
    meta["datetime"] = pd.to_datetime(meta["date_captured"], errors="coerce")

    trans = meta[meta["split"].isin(["trans_test", "trans_val"])]
    train_locs = set(meta[meta["split"] == "train"]["location"])
    if args.location is None:
        cand = trans[~trans["location"].isin(train_locs)]
        div = cand[cand["species"].notna()].groupby("location")["species"].nunique()
        size = cand.groupby("location").size()
        score = size.rank() + div.reindex(size.index).fillna(0).rank()
        loc = str(score.idxmax())
    else:
        loc = str(args.location)
    assert loc not in train_locs, "La cámara de demostración no debe haberse visto en entrenamiento"
    cam_meta = trans[trans["location"] == loc]
    print(f"[cámara] {loc}: {len(cam_meta)} fotos (no vista en entrenamiento)")

    base_thr = json.loads((res / "operating_threshold.json").read_text())["threshold"] \
        if (res / "operating_threshold.json").exists() else mcfg["conf_default"]

    # Inferencia sobre TODAS las fotos de la cámara (como en un despliegue real)
    out_dir = res / "monitor" / f"cam_{loc}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_dir / "detections.csv"
    if pred_csv.exists():
        det = pd.read_csv(pred_csv, dtype={"image_id": str})
    else:
        from ultralytics import YOLO
        model = YOLO(str(p(cfg, "runs") / args.exp / "weights" / "best.pt"))
        rows = []
        for i in range(0, len(cam_meta), 32):
            chunk = cam_meta.iloc[i:i + 32]
            for iid, r in zip(chunk["image_id"], model.predict(list(chunk["path"]), conf=0.05, verbose=False,
                                                               imgsz=cfg["train"]["imgsz"])):
                for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                    rows.append((iid, model.names[int(k)], x1, y1, x2, y2, c))
        det = pd.DataFrame(rows, columns=["image_id", "category", "x1", "y1", "x2", "y2", "score"])
        det.to_csv(pred_csv, index=False)

    f_pred, cnt_pred = frame_table(cam_meta, det, mcfg, base_thr)
    rep_pred = build_report(f_pred, cnt_pred, mcfg)

    # Verdad de campo con el MISMO motor de reglas (incluye animales sin caja)
    gt_cnt = []
    b = boxes[boxes["image_id"].isin(cam_meta["image_id"])]
    gt_cnt.append(b.groupby(["image_id", "category"]).size().rename("n").reset_index())
    nobox = cam_meta[cam_meta["status"] == "animal_nobox"]
    for r in nobox.itertuples():
        for s in str(r.species).split("|"):
            gt_cnt.append(pd.DataFrame([dict(image_id=r.image_id, category=s, n=1)]))
    gcnt = pd.concat(gt_cnt, ignore_index=True)
    gcnt["conf"] = 1.0
    f_gt = cam_meta[["image_id", "location", "datetime", "seq_id"]].copy()
    f_gt["con_fauna"] = cam_meta["status"].ne("empty").to_numpy()
    rep_gt = build_report(f_gt, gcnt, mcfg)

    # ---------------- salidas
    cmp_cam = pd.DataFrame({"predicho": rep_pred["camera"], "real": rep_gt["camera"]})
    sp = rep_pred["species"].add_suffix(" pred").join(rep_gt["species"].add_suffix(" real"), how="outer").fillna(0)
    sp = sp[[c for k in ["eventos", "RAI", "pct_nocturno"] for c in (f"{k} pred", f"{k} real")]]
    cmp_cam.to_csv(out_dir / "resumen_camara.csv")
    sp.to_csv(out_dir / "especies.csv")
    rep_pred["periods"].to_csv(out_dir / "actividad_por_periodo.csv")
    rep_pred["events"].to_csv(out_dir / "eventos_independientes.csv", index=False)
    to_latex(cmp_cam.astype(str), tab / "monitor_camera.tex",
             f"Reporte de monitoreo de la cámara {loc} (no vista en entrenamiento): reglas aplicadas a las "
             "predicciones vs. a la verdad de campo.", "tab:monitorcam", index=True)
    to_latex(sp, tab / "monitor_species.tex",
             f"Actividad por especie en la cámara {loc}: eventos independientes (30 min), RAI "
             "(eventos por 100 días-trampa) y \\% de eventos nocturnos.", "tab:monitorsp", float_fmt="{:.1f}", index=True)

    # Figura: actividad por periodo (pred vs real) y patrón horario
    f_, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for lab, rep, ls in [("predicho", rep_pred, "-"), ("real", rep_gt, "--")]:
        per = rep["periods"]
        if not per.empty:
            per.sum(axis=1).plot(ax=axes[0], ls=ls, marker="o", label=lab)
        ev = rep["events"]
        if not ev.empty:
            ev["start"].dt.hour.value_counts().reindex(range(24), fill_value=0).plot(ax=axes[1], ls=ls, label=lab)
    axes[0].set_title("eventos independientes por periodo"); axes[0].legend(frameon=False)
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].set_title("actividad por hora del día"); axes[1].set_xlabel("hora")
    plt.tight_layout()
    fig.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig / "monitor_activity.pdf"); plt.savefig(fig / "monitor_activity.png", dpi=150)
    plt.close()

    md = [f"# Reporte de monitoreo — cámara {loc}", "",
          f"Modelo: `{args.exp}` · umbral base {base_thr:.2f} · independencia {mcfg['independence_minutes']} min", "",
          "## Resumen de la cámara", cmp_cam.to_markdown(), "",
          "## Por especie", sp.round(1).to_markdown(), "",
          "## Eventos independientes por periodo (predicho)",
          rep_pred["periods"].to_markdown() if not rep_pred["periods"].empty else "_sin eventos_"]
    (out_dir / "reporte.md").write_text("\n".join(md), encoding="utf-8")
    save_json({"location": loc, "camera_pred": rep_pred["camera"], "camera_gt": rep_gt["camera"]},
              res / "monitor_summary.json")
    print("\n".join(md))


if __name__ == "__main__":
    main()
