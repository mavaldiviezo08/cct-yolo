# Monitoreo de fauna silvestre con cámaras trampa (YOLOv8 · CCT-20)

Examen parcial — Redes Neuronales y Aprendizaje Profundo (UNI FIIS, 2026-II) · **Pregunta 3** · Grupo 4

Detector YOLOv8 de especies en fotos de cámaras trampa del benchmark **Caltech Camera Traps (CCT-20)**,
con experimento de generalización a cámaras nunca vistas (*cis* vs *trans*), análisis por oclusión y
tamaño, Grad-CAM y test de enmascarado, y un módulo de monitoreo ecológico basado en reglas.

## Requisitos

- Python 3.10 o superior y GPU NVIDIA para entrenar (probado en Google Colab con T4).
- ~15 GB de disco libres (imágenes de CCT-20: ~6 GB comprimidas).
- Entorno: `requirements.txt`.

## Ejecución (un solo comando)

```bash
pip install -r requirements.txt
bash run_all.sh
```

Este comando reproduce todos los resultados del informe: descarga CCT-20 desde LILA BC, lo convierte a
YOLO, hace el análisis exploratorio, entrena YOLOv8s (30 épocas, ~1.5 h en T4), evalúa en cis/trans,
corre robustez, Grad-CAM, enmascarado y el módulo de monitoreo, y deja las tablas y figuras en
`report/tables/` y `report/figures/`. El informe se compila con `cd report && latexmk -pdf main.tex`
(con `neurips_2025.sty` en `report/` usa el estilo oficial; si no, una réplica del formato).

Verificación rápida sin GPU ni descarga (datos sintéticos, ~10 min en CPU):

```bash
bash run_all.sh --smoke
```

Opciones: `EPOCHS=20 bash run_all.sh`, `MODEL=yolov8n.pt bash run_all.sh`, y
`EXPS="baseline rfs" bash run_all.sh` para entrenar además el experimento con Repeat Factor Sampling
(implementado pero no incluido en el informe por límite de cómputo).

**Google Colab:** `colab_run.ipynb` ejecuta los mismos pasos y guarda pesos y resultados en Drive.

**Modelo entrenado:** `best.pt` (YOLOv8s, 22 MB) está en la sección *Releases* de este repositorio.

**Informe:** `report/Informe_P3_Grupo4.pdf` (formato NeurIPS). Su fuente LaTeX está en `report/`, con las tablas y figuras generadas por el pipeline en `report/tables/` y `report/figures/`.

## Resultados principales

| | Cámaras vistas (cis) | Cámaras nuevas (trans) |
|---|---|---|
| mAP@0.5 | 0.720 | 0.455 |
| mAP@0.5:0.95 | 0.484 | 0.310 |
| mAP@0.5 especies raras | 0.499 | 0.154 |
| Recall "¿hay fauna?" (umbral 0.475) | 0.919 | 0.847 |
| Falsos positivos en fotos vacías | 6.4 % | 9.6 % |
| Detección persiste al borrar el animal | 16.3 % | 9.0 % |

## Estructura

| Paso | Script | Entregable del examen |
|---|---|---|
| Descarga CCT-20 desde LILA BC | `src/download_data.py` | — |
| COCO Camera Traps → YOLO, manejo de vacías y animales sin caja | `src/prepare_yolo.py` | 1 |
| Desbalance de especies, % vacías, tamaños | `src/eda.py` | 1, 2 |
| Entrenamiento | `src/train.py` | 1 |
| mAP@0.5 y mAP@0.5:0.95 por especie, cis vs trans, umbral operativo | `src/evaluate.py` | 2, 3 |
| Oclusión (proxy) y tamaño | `src/robustness.py` | 4 |
| Grad-CAM + test de enmascarado del animal | `src/gradcam.py` | 3 |
| Reporte de actividad por cámara, especie y periodo | `src/monitor.py` | 5 |
| Evaluador AP con cajas ignoradas | `src/metrics.py` (+ `tests/test_metrics.py`) | 2, 4 |
| Informe NeurIPS (incluye discusión de IA responsable) | `report/main.tex` | 6 |

Toda la configuración (rutas, hiperparámetros, aumento de datos, reglas de monitoreo) está en
`configs/config.yaml`.

## Decisiones clave

- **Partición:** `eccv_train` (LILA) = `train` + `cis_*`; `eccv_val` = `trans_*`. Se entrena en `train`,
  se elige el umbral en `cis_val` y se reporta en `cis_test` (cámaras vistas) y `trans_test` (nuevas).
- **Fotos vacías:** el pipeline las incluye como imágenes de fondo con etiqueta vacía (10 % por defecto).
  La partición `train` de CCT-20 no contiene fotos vacías, así que en la práctica el fondo se aprende
  de forma implícita (en YOLOv8 toda región sin caja es un negativo); las vacías de validación y prueba
  se usan para fijar el umbral y medir falsos positivos.
- **Animales sin caja:** fuera del entrenamiento de detección (se aprenderían como fondo); sí entran al monitoreo.
- **Aumento:** mosaico, escala, traslación, volteo horizontal, HSV (noche IR); sin rotación ni volteo vertical.
- **Oclusión:** CCT no la etiqueta. Proxy automático (truncamiento + densidad de bordes). Para validarlo,
  `robustness.py --make-sheet` genera `results/occlusion_sheet/index.html` y la plantilla
  `results/occlusion_manual_TEMPLATE.csv`; al completarla y guardarla como `results/occlusion_manual.csv`,
  `python src/robustness.py` reporta el kappa entre proxy y anotación.

## Datos

Caltech Camera Traps — Beery, Van Horn & Perona, *Recognition in Terra Incognita*, ECCV 2018.
Licencia: Community Data License Agreement (permissive). https://lila.science/datasets/caltech-camera-traps
