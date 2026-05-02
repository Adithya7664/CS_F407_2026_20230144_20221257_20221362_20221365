import random
import torch

try:
    import torch_directml
    DEVICE = torch_directml.device()
    print("Using DirectML GPU")
except ImportError:
    try:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except AssertionError:
        DEVICE = torch.device("cpu")
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
NUM_WORKERS     = 0
VAL_SPLIT       = 0.2          # fraction of training data used for validation
 
# ── Dataset paths (update to your local paths) ───────────────────────────────
import os
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT    = os.path.join(_SRC_DIR, "training_set")
TRAIN_IMAGE_DIR = os.path.join(_SRC_DIR, "training_set", "images")
TRAIN_MASK_DIR  = os.path.join(_SRC_DIR, "training_set", "gt", "semantic", "label_images")
CHECKPOINT_PATH     = "best_model.pth"
 
# ── Number of segmentation classes ───────────────────────────────────────────
NUM_CLASSES = 23          # class IDs 0 – 22
 
# ── Safe landing class IDs (used to build binary safe mask) ──────────────────
# 1 = paved area  |  3 = low vegetation / grass  |  4 = dirt / gravel
SAFE_CLASSES = [1, 2, 3, 4]
 
# ── Class ID → terrain type mapping (for knowledge graph lookup) ──────────────
CLASS_TO_TERRAIN = {
    1: "Pavement",   # paved-area
    2: "Dirt",       # dirt
    3: "Grass",      # grass
    4: "Dirt",       # gravel → treat same as Dirt
}
 
# ── 23-class label names (index = class ID) ───────────────────────────────────
# Source: TU Graz Semantic Drone Dataset label definition
CLASS_NAMES = [
    "unlabeled",    # 0
    "paved-area",   # 1  <- SAFE (Pavement)
    "dirt",         # 2  <- SAFE (Dirt)
    "grass",        # 3  <- SAFE (Grass)
    "gravel",       # 4  <- SAFE (Dirt category)
    "water",        # 5
    "rocks",        # 6
    "pool",         # 7
    "vegetation",   # 8
    "roof",         # 9
    "wall",         # 10
    "window",       # 11
    "door",         # 12
    "fence",        # 13
    "fence-pole",   # 14
    "person",       # 15
    "dog",          # 16
    "car",          # 17
    "bicycle",      # 18
    "tree",         # 19
    "bald-tree",    # 20
    "ar-marker",    # 21
    "obstacle",     # 22
    "conflicting",  # 23
]
 
# ── RGB colour palette for each class (for mask visualisation) ────────────────
# Each entry: (R, G, B)  — matches the dataset's RGB annotation scheme
CLASS_COLORS = [
    (0,   0,   0),      # 0  unlabeled
    (128, 64,  128),    # 1  paved-area    <- SAFE
    (130, 76,  0),      # 2  dirt          <- SAFE
    (0,   102, 0),      # 3  grass         <- SAFE
    (112, 103, 87),     # 4  gravel        <- SAFE
    (28,  42,  168),    # 5  water
    (48,  41,  30),     # 6  rocks
    (0,   50,  89),     # 7  pool
    (107, 142, 35),     # 8  vegetation
    (70,  70,  70),     # 9  roof
    (102, 102, 156),    # 10 wall
    (254, 228, 12),     # 11 window
    (254, 148, 12),     # 12 door
    (190, 153, 153),    # 13 fence
    (153, 153, 153),    # 14 fence-pole
    (255, 22,  96),     # 15 person
    (102, 51,  0),      # 16 dog
    (9,   143, 150),    # 17 car
    (119, 11,  32),     # 18 bicycle
    (51,  51,  0),      # 19 tree
    (190, 250, 190),    # 20 bald-tree
    (112, 150, 146),    # 21 ar-marker
    (2,   135, 115),    # 22 obstacle
    (255, 0,   0),      # 23 conflicting
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