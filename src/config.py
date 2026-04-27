import random
import torch

DEVICE =torch.device("cuda" if torch.cuda.is_available else "cpu")
# ── Image dimensions ──────────────────────────────────────────────────────────
# Raw images are resized to this before ANY processing (geometry anchor size)
MAIN_W = 800
MAIN_H = 600
 
# ANN inference resolution (downscaled from MAIN before feeding the model)
ANN_W = 256
ANN_H = 256
 
# ── Training hyper-parameters ─────────────────────────────────────────────────
NUM_EPOCHS      = 50
BATCH_SIZE      = 8
LEARNING_RATE   = 1e-3
NUM_WORKERS     = 4
VAL_SPLIT       = 0.2          # fraction of training data used for validation
 
# ── Dataset paths (update to your local paths) ───────────────────────────────
DATASET_ROOT        = "data/semantic_drone_dataset"
TRAIN_IMAGE_DIR     = f"{DATASET_ROOT}/original_images"
TRAIN_MASK_DIR      = f"{DATASET_ROOT}/RGB_color_image_masks"
CHECKPOINT_PATH     = "best_model.pth"
 
# ── Number of segmentation classes ───────────────────────────────────────────
NUM_CLASSES = 23          # class IDs 0 – 22
 
# ── Safe landing class IDs (used to build binary safe mask) ──────────────────
# 1 = paved area  |  3 = low vegetation / grass  |  4 = dirt / gravel
SAFE_CLASSES = [1, 3, 4]
 
# ── Class ID → terrain type mapping (for knowledge graph lookup) ──────────────
CLASS_TO_TERRAIN = {
    1: "Pavement",
    3: "Grass",
    4: "Dirt",
}
 
# ── 23-class label names (index = class ID) ───────────────────────────────────
# Source: TU Graz Semantic Drone Dataset label definition
CLASS_NAMES = [
    "unlabeled",          # 0
    "paved-area",         # 1  ← SAFE
    "rocks",              # 2
    "low-vegetation",     # 3  ← SAFE (grass)
    "high-vegetation",    # 4  ← SAFE (dirt/gravel — dataset uses index 4)
    "building",           # 5  (NOTE: verify exact ordering against dataset README)
    "wall",               # 6
    "obstacle",           # 7
    "water",              # 8
    "person",             # 9
    "dog",                # 10
    "car",                # 11
    "bicycle",            # 12
    "tree",               # 13
    "bald-tree",          # 14
    "ar-marker",          # 15
    "obstacle-2",         # 16
    "conflicting",        # 17
    "dirt",               # 18
    "gravel",             # 19
    "grass",              # 20
    "fence",              # 21
    "roof",               # 22
]
 
# ── RGB colour palette for each class (for mask visualisation) ────────────────
# Each entry: (R, G, B)  — matches the dataset's RGB annotation scheme
CLASS_COLORS = [
    (0,   0,   0),      # 0  unlabeled        — black
    (128, 64,  128),    # 1  paved-area        — purple
    (130, 76,  0),      # 2  rocks             — brown
    (0,   102, 0),      # 3  low-vegetation    — dark green
    (112, 103, 87),     # 4  high-vegetation   — olive
    (28,  42,  168),    # 5  building          — blue
    (48,  41,  30),     # 6  wall              — dark brown
    (0,   50,  89),     # 7  obstacle          — dark teal
    (107, 142, 35),     # 8  water             — yellow-green
    (70,  70,  70),     # 9  person            — grey
    (102, 102, 156),    # 10 dog               — slate
    (190, 153, 153),    # 11 car               — pink
    (9,   143, 150),    # 12 bicycle           — cyan
    (119, 11,  32),     # 13 tree              — dark red
    (0,   0,   142),    # 14 bald-tree         — navy
    (0,   0,   90),     # 15 ar-marker         — deep blue
    (0,   0,   230),    # 16 obstacle-2        — bright blue
    (119, 11,  32),     # 17 conflicting       — dark red
    (0,   60,  100),    # 18 dirt              — dark teal
    (0,   0,   142),    # 19 gravel            — navy
    (0,   80,  100),    # 20 grass             — teal
    (128, 64,  255),    # 21 fence             — violet
    (0,   0,   192),    # 22 roof              — medium blue
]
 
# ── Cost function weights ─────────────────────────────────────────────────────
W_DISTANCE = 0.4
W_ROUGHNESS = 0.2
W_SEMANTIC  = 0.4
 
# ── Safe zone search parameters ───────────────────────────────────────────────
MAX_SLOPE       = 0.15      # candidates with mean Sobel >= this are discarded
BOX_W           = 80        # landing bounding box width  (pixels at 800×600)
BOX_H           = 80        # landing bounding box height (pixels at 800×600)
SEARCH_STEP     = 20        # grid search stride (pixels)
ANGLE_STEPS     = [0, 10, 20, 30, 45]   # rotation angles to try (degrees)
N_INTERIOR_PTS  = 50        # number of interior sample points per candidate box
 
# ── MiDaS model name (loaded via torch.hub) ───────────────────────────────────
MIDAS_MODEL     = "MiDaS_small"
MIDAS_REPO      = "intel-isl/MiDaS"
 
# ── Mission config path ───────────────────────────────────────────────────────
MISSION_CONFIG_PATH = "mission_config.json"
 
# ── Output file names ─────────────────────────────────────────────────────────
OUTPUT_IMAGE    = "output.jpg"
OUTPUT_ANALYSIS = "output_analysis.jpg"