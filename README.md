# CS_F407_2026_20230144_20221257_20221362_20221365
# AI assignment 2: — Drone Landing Zone AI

> An end-to-end AI system that autonomously identifies the safest and closest landing zone for a package drop from a single downward-facing drone camera frame, using semantic segmentation, monocular depth estimation, and a graph-based semantic reasoning engine.

---

## Group Details

| Member | ID |
|---|---|
| Meduri Adithya | 20230144 |
| Drishti Chatterjee | 20221257 |
| Utkarsh anand | 20221362 |
| Riya Kasliwal | 20221365 |

---

## Project Overview

The system takes a single aerial drone image and a user-specified target coordinate, then outputs the optimal package drop location by:

1. **Semantic Segmentation** — A custom U-Net with pretrained ResNet18 encoder (4 encoder + 4 decoder layers) predicts per-pixel class labels across 24 terrain classes
2. **Depth Estimation** — Intel MiDaS (MiDaS_small) produces a relative depth map to measure terrain roughness/slope
3. **Safe Zone Search** — A rotational grid search finds candidate bounding boxes that lie entirely on safe terrain classes (paved-area, dirt, grass, gravel) and pass a slope filter
4. **Knowledge Graph** — A 3-layer spreading activation graph computes terrain penalty scores based on the package's traits (fragile, heavy, valuable, biohazard)
5. **Cost Ranking** — Candidates are ranked by a weighted cost function combining distance, roughness, and semantic penalty

---

## Repository Structure

```
src/
├── config.py              # All constants — image sizes, class labels, cost weights
├── dataset.py             # GrazDataset class — loads images + RGB masks
├── model.py               # SegNet — U-Net with pretrained ResNet18 encoder
├── train.py               # Training loop — CrossEntropy + Adam, saves best_model.pth
├── geometry.py            # Safe mask, rotational grid search, cost scoring
├── knowledge_graph.py     # 3-layer spreading activation graph, think() function
├── semantic_brain.py      # Terrain classifier + mission_config.json parser
├── utils.py               # Colour mapping, drawing, dashboard generation
├── main.py                # Entry point — full inference pipeline
├── mission_config.json    # Package traits for knowledge graph
├── requirements.txt       # Python dependencies
├── best_model.pth         # Trained model checkpoint (download from Drive above)
└── training_set/          # TU Graz Semantic Drone Dataset (not tracked in git)
    ├── images/            # 400 aerial drone images
    └── gt/semantic/
        ├── label_images/  # RGB annotation masks
        └── class_dict.csv # Colour palette definition
```

---

## Dataset

**TU Graz Semantic Drone Dataset** — 400 aerial drone images at 6000×4000px acquired at 5–30m altitude above ground, with dense semantic annotations across 24 classes.

Download: https://cloud.tugraz.at/index.php/s/csxYfaKmie6LyqA
Extract and place the `training_set/` folder inside `src/`.

### Safe Landing Classes

| Class ID | Name | Terrain Type |
|---|---|---|
| 1 | paved-area | Pavement |
| 2 | dirt | Dirt |
| 3 | grass | Grass |
| 4 | gravel | Dirt |

---

## Environment Setup

### Prerequisites
- Python 3.10+
- Conda (Miniconda or Anaconda)

### Create Conda Environment
```bash
conda env create -f config.yml
conda activate CS_F407_2026_20230144_20221257_20221362_20221365
```

### Or install via pip
```bash
pip install -r requirements.txt
```

### For Intel Arc GPU on Windows (optional)
```bash
pip install torch-directml
```

### Training Results

| Metric | Value |
|---|---|
| Epochs trained | 20 |
| Final Train Loss | ~1.5 |
| Final Val Loss | ~1.7 |
| Final Val Accuracy | ~55% |
| Training Hardware | Intel Arc A370M (DirectML) |

---

## Inference

### Interactive mode
```bash
cd src
python main.py
```
The system will prompt:
```
Enter image file path: Copy the absolute file path of the image for testing and paste and run
Enter target X coordinate (0–800): enter the x coordinate for the target
Enter target Y coordinate (0–600): enter the y coordinate
```

### Mission Config
Edit `mission_config.json` to set package traits before running:
```json
{
    "mission_id": "OP-DELTA-9",
    "package": {
        "type": "medical_vials",
        "heavy": true,
        "fragile": false,
        "valuable": false,
        "biohazard": false
    }
}
```
Ignore if it is already there
---

## Outputs

Two files are generated in `src/` after each run:

### `output.jpg`
Original 800×600 image with:
- 🟥 Red crosshair at user-specified target coordinate
- 🟩 Green oriented bounding box at optimal landing zone
- Cost label with terrain type and total score

### `output_analysis.jpg`
6-panel analysis dashboard:

| Panel | Content |
|---|---|
| Top-left | Original image (800×600) |
| Top-center | Semantic segmentation (U-Net output) |
| Top-right | Depth map (MiDaS colourised) |
| Bottom-left | Binary safe zone mask |
| Bottom-center | Best placement with bounding box |
| Bottom-right | Cost breakdown panel |

---

## System Architecture

### Pipeline (in order)
```
Input image
    ↓
Resize to 800×600  (MAIN resolution)
    ↓
Downscale to 256×256  →  SegNet  →  Argmax  →  Upscale to 800×600
    ↓                                                    ↓
MiDaS depth map (800×600)                    Semantic seg map (800×600)
    ↓                                                    ↓
                    Binary safe mask (classes 1,2,3,4)
                            ↓
                    Rotational grid search
                            ↓
                    Slope filter (Sobel < 0.15)
                            ↓
                    Terrain classification per candidate
                            ↓
                    Knowledge graph → semantic penalty
                            ↓
                    Cost = 0.4×distance + 0.2×roughness + 0.4×semantic
                            ↓
                    Best candidate → green bounding box
```

### Cost Function
```
Cost = 0.4 × distance + 0.2 × roughness + 0.4 × semantic_penalty

distance         = √((cx−xt)² + (cy−yt)²) / 1000
roughness        = mean_sobel_magnitude / 0.15
semantic_penalty = think(active_nodes)[terrain_class]  ∈ [0.0, 1.0]
```

### Knowledge Graph (3-layer Spreading Activation)
```
Layer 1 — Source Nodes:   Fragile, Valuable, Biohazard, Heavy
              ↓ (weighted edges)
Layer 2 — Property Nodes: Hard, Wet, Slippery, Dirty, Visible,
                           Contaminated, Soft, Unstable
              ↓ (weighted edges)
Layer 3 — Terrain Nodes:  Pavement, Grass, Dirt
```
All penalty scores derived solely from edge-weight accumulation — no if/else chains.

---

## Sample Results
Sample results are already in the src file for a particular image
---

## Known Limitations

- Model trained for only 20 epochs due to hardware constraints (Intel Arc A370M, 4GB VRAM)
- Batch size limited to 2 due to VRAM — causes noisy gradient updates
- DirectML backend (Windows Intel GPU) has some operator fallbacks to CPU
- 55% pixel accuracy is sufficient for demonstrating the end-to-end pipeline

---
