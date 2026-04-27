import math
import cv2
import numpy as np
import torch
 
import config

#1. Binary safe zone mask
def build_safe_mask(seg_map: np.ndarray) -> np.ndarray:
    
    # Convert a (H, W) integer class-ID map (at 800×600) into a binary
    # uint8 mask:  255 = safe pixel,  0 = unsafe pixel.
 
    # Safe classes (from config.SAFE_CLASSES): 1, 3, 4
    # (paved-area, low-vegetation/grass, dirt/gravel)
 
    # Parameters
    # ----------
    # seg_map : np.ndarray  shape (H, W),  dtype int  — class IDs at 800×600
 
    # Returns
    # -------
    # mask : np.ndarray  shape (H, W),  dtype uint8
    
    mask = np.zeros(seg_map.shape, dtype=np.uint8)
    for cls_id in config.SAFE_CLASSES:
        mask[seg_map == cls_id] = 255
    return mask

#Rotated bounding box utilities
def _box_corners(cx: float, cy: float,
                 w: int, h: int,
                 angle_deg: float) -> np.ndarray:
    """
    Return the four corners of a rotated rectangle as (4, 2) float32 array.
 
    Parameters
    ----------
    cx, cy     : centre of the box (pixels)
    w, h       : box width and height (pixels)
    angle_deg  : rotation angle in degrees (counter-clockwise)
    """
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
 
    hw, hh = w / 2, h / 2
    # Un-rotated corner offsets relative to centre
    offsets = np.array([
        [-hw, -hh],
        [ hw, -hh],
        [ hw,  hh],
        [-hw,  hh],
    ], dtype=np.float32)
 
    rot = np.array([[cos_t, -sin_t],
                    [sin_t,  cos_t]], dtype=np.float32)
    return (offsets @ rot.T) + np.array([cx, cy], dtype=np.float32)
 
 
def _box_fully_inside_image(corners: np.ndarray,
                             img_w: int, img_h: int) -> bool:
    """True if all four corners lie within [0, img_w) × [0, img_h)."""
    return (
        corners[:, 0].min() >= 0 and
        corners[:, 0].max() <  img_w and
        corners[:, 1].min() >= 0 and
        corners[:, 1].max() <  img_h
    )
 
 
def _interior_sample_points(cx: float, cy: float,
                             w: int, h: int,
                             angle_deg: float,
                             n: int = config.N_INTERIOR_PTS) -> np.ndarray:
    """
    Sample `n` points uniformly distributed inside a rotated rectangle.
 
    Strategy: sample on a regular grid in the un-rotated frame, then
    rotate them into image space. Guarantees all points are interior.
 
    Returns
    -------
    pts : np.ndarray  shape (n, 2)  — (x, y) pixel coordinates (float32)
    """
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rot = np.array([[cos_t, -sin_t],
                    [sin_t,  cos_t]], dtype=np.float32)
 
    side = max(1, int(math.sqrt(n)))
    # Slightly shrink to keep points strictly interior (avoid edge pixels)
    xs = np.linspace(-w / 2 * 0.85, w / 2 * 0.85, side)
    ys = np.linspace(-h / 2 * 0.85, h / 2 * 0.85, side)
    grid = np.array([[x, y] for x in xs for y in ys], dtype=np.float32)
 
    rotated = grid @ rot.T + np.array([cx, cy], dtype=np.float32)
    return rotated[:n]

#sobel roughness

def _compute_sobel_map(depth_map: np.ndarray) -> np.ndarray:
    """
    Compute normalised Sobel gradient magnitude of the depth map.
 
    Parameters
    ----------
    depth_map : np.ndarray  shape (H, W),  uint8 [0–255]
 
    Returns
    -------
    sobel : np.ndarray  shape (H, W),  float32 in [0, 1]
    """
    depth_f = depth_map.astype(np.float32) / 255.0
    gx = cv2.Sobel(depth_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    # Normalise to [0, 1]
    mag_max = mag.max()
    if mag_max > 0:
        mag /= mag_max
    return mag
 
 
def _mean_roughness_at_box(sobel_map: np.ndarray,
                            pts: np.ndarray) -> float:
    """
    Average Sobel magnitude at interior sample points.
 
    Parameters
    ----------
    sobel_map : (H, W) float32
    pts       : (N, 2) float32  — (x, y) pixel coords
 
    Returns
    -------
    mean roughness : float
    """
    H, W = sobel_map.shape
    values = []
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            values.append(sobel_map[yi, xi])
    return float(np.mean(values)) if values else 1.0   # default max if no valid pts

#safe pixel ratio at a candidate box

def _all_pts_on_safe_mask(safe_mask: np.ndarray,
                           pts: np.ndarray) -> bool:
    """
    Returns True only if every interior sample point falls on a white
    (255) pixel of the binary safe mask.
    """
    H, W = safe_mask.shape
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < W and 0 <= yi < H):
            return False
        if safe_mask[yi, xi] == 0:
            return False
    return True

#rotational grid search
def find_candidate_boxes(safe_mask: np.ndarray, depth_map:  np.ndarray, box_w: int = config.BOX_W,box_h: int = config.BOX_H, step: int = config.SEARCH_STEP, angles: list = config.ANGLE_STEPS,max_slope: float = config.MAX_SLOPE)-> list[dict]:
    """
    Scan the entire 800×600 image with a rotatable bounding box to
    collect every valid candidate landing region.
 
    Validity criteria
    -----------------
    1. Box is fully within image boundaries.
    2. All interior sample points land on white pixels of `safe_mask`.
    3. Mean Sobel roughness under the box < `max_slope`.
 
    Parameters
    ----------
    safe_mask  : (H, W) uint8    — binary safe mask (255 = safe)
    depth_map  : (H, W) uint8    — MiDaS depth map [0–255]
    box_w, h   : landing box size in pixels (at 800×600)
    step       : grid search stride in pixels
    angles     : list of rotation angles to try (degrees)
    max_slope  : roughness threshold; candidates above this are discarded
 
    Returns
    -------
    candidates : list of dicts, each with keys:
        cx, cy      — centre pixel (float)
        angle       — rotation angle (degrees)
        corners     — (4, 2) float32 corners
        roughness   — normalised roughness score [0, 1)
        pts         — interior sample points
    """
    H, W     = safe_mask.shape
    sobel    = _compute_sobel_map(depth_map)
    candidates = []
 
    half_w = box_w // 2
    half_h = box_h // 2
 
    # Grid centres: keep centre far enough from edges to allow rotation
    margin = int(math.ceil(math.sqrt(half_w**2 + half_h**2))) + 1
 
    xs = range(margin, W - margin, step)
    ys = range(margin, H - margin, step)
 
    for cy in ys:
        for cx in xs:
            for angle in angles:
                corners = _box_corners(cx, cy, box_w, box_h, angle)
 
                # 1. Must fit inside image
                if not _box_fully_inside_image(corners, W, H):
                    continue
 
                # 2. All interior points must be on safe pixels
                pts = _interior_sample_points(cx, cy, box_w, box_h, angle)
                if not _all_pts_on_safe_mask(safe_mask, pts):
                    continue
 
                # 3. Roughness filter
                raw_roughness = _mean_roughness_at_box(sobel, pts)
                if raw_roughness >= max_slope:
                    continue
 
                candidates.append({
                    "cx":        float(cx),
                    "cy":        float(cy),
                    "angle":     angle,
                    "corners":   corners,
                    "roughness": raw_roughness / max_slope,   # normalised [0,1)
                    "pts":       pts,
                })
 
    return candidates

#cost scoring and ranking
def compute_costs(
    candidates:      list[dict],
    target_xy:       tuple[float, float],
    semantic_scores: dict[int, float],
    seg_map:         np.ndarray,
) -> list[dict]:
    """
    Attach a scalar cost to every candidate and return them sorted
    lowest → highest cost.
 
    Cost = W_DISTANCE × distance
         + W_ROUGHNESS × roughness
         + W_SEMANTIC  × semantic_penalty
 
    Parameters
    ----------
    candidates      : list from find_candidate_boxes()
    target_xy       : (xt, yt) user-specified target pixel in 800×600 space
    semantic_scores : dict {terrain_class_id: penalty_score}
                      produced by semantic_brain.get_terrain_penalty()
    seg_map         : (H, W) int  class-ID map at 800×600 — used to find
                      dominant terrain class under each candidate
 
    Returns
    -------
    ranked : list of candidate dicts with extra keys:
        distance         — normalised Euclidean distance
        semantic_penalty — knowledge-graph penalty [0, 1]
        cost             — total scalar cost
    """
    xt, yt = target_xy
 
    for cand in candidates:
        cx, cy = cand["cx"], cand["cy"]
 
        # Distance term
        dist = math.sqrt((cx - xt) ** 2 + (cy - yt) ** 2) / 1000.0
 
        # Dominant terrain class under this box
        terrain_cls = _dominant_class(seg_map, cand["pts"])
 
        # Semantic penalty from knowledge graph
        sem_penalty = semantic_scores.get(terrain_cls, 0.5)
 
        cost = (config.W_DISTANCE  * dist +
                config.W_ROUGHNESS * cand["roughness"] +
                config.W_SEMANTIC  * sem_penalty)
 
        cand["distance"]         = dist
        cand["terrain_class"]    = terrain_cls
        cand["semantic_penalty"] = sem_penalty
        cand["cost"]             = cost
 
    return sorted(candidates, key=lambda c: c["cost"])
 
 
def _dominant_class(seg_map: np.ndarray, pts: np.ndarray) -> int:
    """
    Find the most frequent segmentation class among interior sample points.
 
    Returns the mode class ID (int), defaulting to 0 if no valid points.
    """
    H, W = seg_map.shape
    counts = {}
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            cls = int(seg_map[yi, xi])
            counts[cls] = counts.get(cls, 0) + 1
    if not counts:
        return 0
    return max(counts, key=counts.get)

#upscale segmentation mask 256x256->800x600
def upscale_seg_map(seg_256: np.ndarray) -> np.ndarray:
    """
    Upscale the ANN output mask (256×256 int class IDs) back to 800×600
    using nearest-neighbour interpolation to preserve class labels.
 
    Parameters
    ----------
    seg_256 : np.ndarray  shape (256, 256),  dtype int
 
    Returns
    -------
    seg_800 : np.ndarray  shape (600, 800),  dtype int
    """
    seg_uint8 = seg_256.astype(np.uint8)
    seg_800   = cv2.resize(seg_uint8,
                           (config.MAIN_W, config.MAIN_H),
                           interpolation=cv2.INTER_NEAREST)
    return seg_800.astype(np.int32)

if __name__ == "__main__":
    print("Running geometry.py sanity check…")
 
    # Synthetic safe mask — central 400×300 region is safe
    mask = np.zeros((config.MAIN_H, config.MAIN_W), dtype=np.uint8)
    mask[150:450, 200:600] = 255
 
    # Flat depth map (no roughness)
    depth = np.full((config.MAIN_H, config.MAIN_W), 128, dtype=np.uint8)
 
    candidates = find_candidate_boxes(mask, depth)
    print(f"  Found {len(candidates)} candidates")
    assert len(candidates) > 0, "Expected at least one candidate in safe region!"
 
    # Dummy seg map (all class 1 = pavement)
    seg = np.ones((config.MAIN_H, config.MAIN_W), dtype=np.int32)
    semantic_scores = {1: 0.3, 3: 0.1, 4: 0.5}
 
    ranked = compute_costs(candidates, (400, 300), semantic_scores, seg)
    best = ranked[0]
    print(f"  Best candidate: cx={best['cx']:.0f}, cy={best['cy']:.0f}, "
          f"cost={best['cost']:.4f}")
 

