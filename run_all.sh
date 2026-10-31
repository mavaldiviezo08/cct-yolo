#!/usr/bin/env bash
# Pipeline completo de la Pregunta 3 con un único comando.
#   bash run_all.sh            -> datos reales CCT-20 (requiere GPU, ~2.5 h: descarga + entrenamiento + evaluación)
#   bash run_all.sh --smoke    -> prueba de punta a punta con datos sintéticos (CPU, ~10 min)
# Variables opcionales: EXPS="baseline rfs" EPOCHS=30 MODEL=yolov8s.pt
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--smoke" ]]; then
  export CCT_RAW=data_smoke/raw CCT_IMAGES=data_smoke/raw/eccv_18_all_images_sm \
         CCT_YOLO=data_smoke/yolo CCT_RUNS=runs_smoke CCT_RESULTS=results_smoke \
         CCT_REPORT_TABLES=results_smoke/tables CCT_REPORT_FIGURES=results_smoke/figures
  EXPS="${EXPS:-baseline}"; EPOCHS="${EPOCHS:-3}"; MODEL="${MODEL:-yolov8n.pt}"
  EXTRA="--imgsz 320 --batch 8"
  python tests/make_synthetic.py "$CCT_RAW"
else
  EXPS="${EXPS:-baseline}"; EPOCHS="${EPOCHS:-}"; MODEL="${MODEL:-}"
  EXTRA=""
  python src/download_data.py
fi

python src/prepare_yolo.py
python src/eda.py
for exp in $EXPS; do
  python src/train.py --exp "$exp" ${EPOCHS:+--epochs $EPOCHS} ${MODEL:+--model $MODEL} $EXTRA
done
python src/evaluate.py --main baseline
python src/robustness.py --exp baseline --make-sheet
python src/gradcam.py --exp baseline
python src/monitor.py --exp baseline

if [[ "${1:-}" != "--smoke" ]] && command -v latexmk >/dev/null; then
  (cd report && latexmk -pdf -interaction=nonstopmode main.tex >/dev/null && echo "[informe] report/main.pdf")
fi
echo "[listo] métricas en results/, tablas y figuras en report/"
