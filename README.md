# Ship Detection in Aerial Imagery

Deep-learning-based **single-class ship detection** in aerial imagery using YOLOv5nu and YOLOv8n, with preprocessing, quantitative evaluation, occlusion analysis, and saliency-based interpretability.

## Overview
This project studies ship detection in challenging aerial and satellite imagery using two YOLO architectures: **YOLOv5nu** for lightweight real-time inference and **YOLOv8n** for higher-accuracy detection.

## Pipeline
```text
Aerial Images → Auto-Orientation → Center Crop (25–75%)
→ Resize 640×640 → Normalization → Train/Validation/Test
→ YOLOv5nu / YOLOv8n → Detection → Evaluation
→ Occlusion & Saliency Analysis
```

## Dataset
According to the accompanying report:
- **9,616 images**
- Single class: **ship**
- Drone and satellite imagery
- Resolution range: approximately **250×200 to 3000×1440**
- Bounding-box annotations
- Split: **71% train / 16% validation / 13% test**

The dataset is not included in this repository.

## Models
- **YOLOv5nu** — lightweight model intended for real-time/resource-constrained deployment.
- **YOLOv8n** — anchor-free YOLO architecture with C2f feature extraction, multi-scale feature fusion, and a decoupled detection head.

## Results

| Model | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| YOLOv5nu | 0.67945 | 0.49474 | 0.58786 | 0.3594 |
| YOLOv8n | 0.71560 | 0.53008 | 0.61228 | 0.3781 |

### Training Loss

| Model | Box Loss | Classification Loss | DFL Loss |
|---|---:|---:|---:|
| YOLOv5nu | 0.91453 | 0.84882 | 1.21857 |
| YOLOv8n | 0.86967 | 0.65627 | 1.17463 |

The reported evaluation shows higher precision, recall, mAP@50 and mAP@50–95 for YOLOv8n under the evaluated setup.

## Interpretability
The project includes:
- Occlusion Sensitivity Maps
- Saliency Maps

These analyses examine which image regions contribute to predictions. The report describes more concentrated relevance around ship structures for YOLOv8n, while YOLOv5nu showed more diffuse attention in challenging scenes.

## Preprocessing
1. Auto-orientation using image metadata
2. Center cropping to the 25–75% region
3. Resize to 640×640
4. Normalize pixel values
5. Train/validation/test split

## Repository Structure
```text
Ship-Detection-in-Aerial-Imagery/
├── README.md
├── PROJECT_INFO.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── Object_Recognition_in_Aerial_Imagery.ipynb
├── src/
│   └── notebook_code_reference.py
├── results/
│   ├── figures/
│   └── metrics/
└── report/
    └── Ship_Detection_in_Aerial_Imagery_Report.pdf
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Roboflow API Key
The notebook requires a Roboflow API key for dataset access. The GitHub-facing notebook has been sanitized so the key is **not stored in the repository**.

Linux/macOS:
```bash
export ROBOFLOW_API_KEY="YOUR_API_KEY"
```

Windows PowerShell:
```powershell
$env:ROBOFLOW_API_KEY="YOUR_API_KEY"
```

**Never commit an API key to GitHub.**

## Running
Open:
`notebooks/Object_Recognition_in_Aerial_Imagery.ipynb`

The notebook covers dataset access, YOLO training, prediction visualization, occlusion analysis, and saliency analysis.

## Future Work
- Integrate SAR/radar with optical imagery
- Extend from single-class to multi-class maritime detection
- Optimize and quantize models for edge deployment
- Deploy on low-power UAV/edge platforms
