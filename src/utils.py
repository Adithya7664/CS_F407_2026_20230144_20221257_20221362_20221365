import cv2
import numpy as np
from PIL import Image
 
import config

#color palette
# Shape (NUM_CLASSES, 3) uint8  — row i = RGB colour for class i
PALETTE = np.array(config.CLASS_COLORS, dtype=np.uint8)   # (23, 3)

#mask->rgb conversions
def class_ids_to_rgb(seg_map: np.ndarray) -> np.ndarray:
    """
    Convert an (H, W) integer class-ID map to an (H, W, 3) uint8 RGB image
    using the palette defined in config.CLASS_COLORS.
 
    Parameters
    ----------
    seg_map : np.ndarray  shape (H, W),  dtype int
 
    Returns
    -------
    rgb : np.ndarray  shape (H, W, 3),  dtype uint8
    """
    seg_clipped = np.clip(seg_map, 0, len(PALETTE) - 1).astype(np.int32)
    return PALETTE[seg_clipped]
 
 
def rgb_to_class_ids(rgb: np.ndarray) -> np.ndarray:
    """
    Convert an (H, W, 3) uint8 RGB annotation mask back to an (H, W) int
    class-ID map by matching each pixel against the palette.
 
    Unknown colours map to class 0 (unlabeled).
 
    Parameters
    ----------
    rgb : np.ndarray  shape (H, W, 3),  dtype uint8
 
    Returns
    -------
    seg_map : np.ndarray  shape (H, W),  dtype int32
    """
    H, W, _ = rgb.shape
    flat    = rgb.reshape(-1, 3)
 
    # Build lookup: tuple(R,G,B) → class_id
    lut = {tuple(color): idx for idx, color in enumerate(config.CLASS_COLORS)}
 
    out = np.zeros(H * W, dtype=np.int32)
    for i, px in enumerate(flat):
        out[i] = lut.get(tuple(px), 0)
    return out.reshape(H, W)
 
 
def binary_mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """
    Convert a (H, W) uint8 binary mask (0 or 255) to a 3-channel
    RGB image for display purposes.
    """
    return np.stack([mask, mask, mask], axis=-1)

#depth map colorisation
def depth_to_colormap(depth: np.ndarray,
                      colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """
    Convert a uint8 [0–255] depth map to a false-colour RGB image.
 
    Bright (high value) = close to drone.
    Dark  (low  value)  = far from drone.
 
    Parameters
    ----------
    depth    : np.ndarray  shape (H, W),  dtype uint8
    colormap : OpenCV colormap constant  (default COLORMAP_INFERNO)
 
    Returns
    -------
    coloured : np.ndarray  shape (H, W, 3),  dtype uint8  (RGB)
    """
    bgr = cv2.applyColorMap(depth, colormap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

#bounding box drawing
def draw_oriented_box(image: np.ndarray,
                      corners: np.ndarray,
                      color:   tuple = (0, 255, 0),
                      thickness: int = 2) -> np.ndarray:
    """
    Draw a rotated (oriented) bounding box on a copy of `image`.
 
    Parameters
    ----------
    image   : (H, W, 3) uint8 RGB
    corners : (4, 2) float32  — box corners in (x, y) pixel coordinates
    color   : RGB tuple
    thickness: line thickness in pixels
 
    Returns
    -------
    out : (H, W, 3) uint8 RGB  — image with box drawn
    """
    out = image.copy()
    # OpenCV drawContours expects BGR and integer pts
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    pts = corners.reshape((-1, 1, 2)).astype(np.int32)
    bgr_color = (color[2], color[1], color[0])    # RGB → BGR
    cv2.polylines(bgr, [pts], isClosed=True,
                  color=bgr_color, thickness=thickness)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
 
 
def draw_target_marker(image: np.ndarray,
                       xy:    tuple[int, int],
                       color: tuple = (255, 0, 0),
                       radius: int  = 8,
                       thickness: int = 2) -> np.ndarray:
    """
    Draw a crosshair + circle marker at the user-specified target location.
 
    Parameters
    ----------
    image  : (H, W, 3) uint8 RGB
    xy     : (x, y) pixel coordinate of target
    color  : RGB tuple  (default red)
    """
    out = image.copy()
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    bgr_color = (color[2], color[1], color[0])
    cx, cy = int(xy[0]), int(xy[1])
 
    # Circle
    cv2.circle(bgr, (cx, cy), radius, bgr_color, thickness)
    # Crosshair
    arm = radius + 4
    cv2.line(bgr, (cx - arm, cy), (cx + arm, cy), bgr_color, thickness)
    cv2.line(bgr, (cx, cy - arm), (cx, cy + arm), bgr_color, thickness)
 
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
 
 
def draw_cost_label(image: np.ndarray,
                    cx: int, cy: int,
                    cost: float,
                    terrain: str = "",
                    color: tuple = (0, 255, 0)) -> np.ndarray:
    """
    Overlay a small text label near the best bounding box centre.
    """
    out = image.copy()
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    bgr_color = (color[2], color[1], color[0])
 
    label = f"Best: ({cx},{cy}) @ {terrain} | Cost: {cost:.4f}"
    cv2.putText(bgr, label, (max(cx - 80, 5), max(cy - 15, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr_color, 1, cv2.LINE_AA)
 
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

#6 panel analysis dashboard
def build_analysis_dashboard(
    original:   np.ndarray,
    seg_rgb:    np.ndarray,
    depth_rgb:  np.ndarray,
    safe_rgb:   np.ndarray,
    placement:  np.ndarray,
    cost_panel: np.ndarray,
    best_cost:  float = 0.0,
    panel_size: tuple = (267, 200),   # (W, H) per panel
) -> np.ndarray:
    
    # Assemble the 6-panel output_analysis.jpg dashboard.
 
    # Layout (2 rows × 3 columns):
    # ┌──────────────┬──────────────┬──────────────┐
    # │  Original    │  Seg (U-Net) │  Depth (MiDaS│
    # ├──────────────┼──────────────┼──────────────┤
    # │  Safe Mask   │  Best Place  │  Cost Panel  │
    # └──────────────┴──────────────┴──────────────┘
 
    # All panels are resized to `panel_size` before stitching.
 
    # Parameters
    # ----------
    # original  : (H, W, 3) RGB — the 800×600 input image
    # seg_rgb   : (H, W, 3) RGB — colourised segmentation
    # depth_rgb : (H, W, 3) RGB — colourised depth map
    # safe_rgb  : (H, W, 3) RGB — binary safe mask as RGB
    # placement : (H, W, 3) RGB — image with bounding box drawn
    # cost_panel: (H, W, 3) RGB — text cost breakdown panel
 
    # Returns
    # -------
    # dashboard : np.ndarray  (H_total, W_total, 3) uint8
    
    pw, ph = panel_size
 
    def _resize(img):
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (pw, ph))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
 
    panels = [original, seg_rgb, depth_rgb, safe_rgb, placement, cost_panel]
    resized = [_resize(p) for p in panels]
 
    # Add titles
    titles = ["Original Image (800x600)",
              "Semantic Segmentation (U-Net)",
              "Depth Map (MiDaS)",
              "Safe Zone Mask",
              f"Best Placement (Cost: {best_cost:.4f})",
              "Cost Breakdown"]
 
    titled = []
    for img, title in zip(resized, titles):
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, title, (5, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (220, 220, 220), 1, cv2.LINE_AA)
        titled.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
 
    row1 = np.concatenate(titled[:3], axis=1)
    row2 = np.concatenate(titled[3:], axis=1)
    return np.concatenate([row1, row2], axis=0)
 
 
def build_cost_text_panel(best: dict,
                           active_nodes: list[str],
                           panel_size: tuple = (267, 200)) -> np.ndarray:
    """
    Render a plain-text cost breakdown panel as an RGB image.
 
    Parameters
    ----------
    best         : ranked candidate dict (from geometry.compute_costs)
    active_nodes : list of active source node names from mission_config
    panel_size   : (W, H) of the output panel in pixels
 
    Returns
    -------
    panel : (H, W, 3) uint8 RGB
    """
    pw, ph = panel_size
    bg = np.full((ph, pw, 3), 245, dtype=np.uint8)   # light grey background
    bgr = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
 
    terrain_name = config.CLASS_TO_TERRAIN.get(
        best.get("terrain_class", 0), "Unknown"
    )
 
    lines = [
        "BEST PLACEMENT (SEMANTIC REASONING)",
        "",
        f"Position: ({int(best['cx'])}, {int(best['cy'])})",
        f"Angle: {best['angle']}°",
        f"Terrain: {terrain_name}",
        f"Distance: {best['distance']:.4f}px",
        f"Roughness: {best['roughness']:.4f}",
        f"Semantic Penalty: {best['semantic_penalty']:.4f}",
        "",
        "Mission Factors:",
        ", ".join(active_nodes) if active_nodes else "None",
        "",
        "Cost Breakdown:",
        f"Distance ({config.W_DISTANCE}): {config.W_DISTANCE * best['distance']:.4f}",
        f"Roughness ({config.W_ROUGHNESS}): {config.W_ROUGHNESS * best['roughness']:.4f}",
        f"Semantic ({config.W_SEMANTIC}): {config.W_SEMANTIC * best['semantic_penalty']:.4f}",
        f"TOTAL: {best['cost']:.4f}",
    ]
 
    y = 16
    for line in lines:
        is_header = line == lines[0]
        font_scale = 0.32 if not is_header else 0.33
        weight = cv2.FONT_HERSHEY_SIMPLEX
        color  = (20, 20, 20) if not is_header else (10, 10, 120)
        cv2.putText(bgr, line, (6, y), weight, font_scale, color, 1, cv2.LINE_AA)
        y += 11
 
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

#image i/o helpers
def load_image_rgb(path: str) -> np.ndarray:
    """Load any image file as (H, W, 3) uint8 RGB."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)
 
 
def save_image_rgb(path: str, image: np.ndarray):
    """Save an (H, W, 3) uint8 RGB array to disk as JPEG/PNG."""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)
    print(f"[utils] Saved → {path}")
 
 
def resize_to_main(image: np.ndarray) -> np.ndarray:
    """Resize any RGB image to MAIN_W × MAIN_H (800×600)."""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (config.MAIN_W, config.MAIN_H))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
 
 
def tensor_to_numpy_image(tensor) -> np.ndarray:
    """
    Convert a (3, H, W) float32 torch tensor (normalised) back to
    (H, W, 3) uint8 RGB for display.
    """
    import torch
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
 
    # Undo ImageNet normalisation
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = tensor.transpose(1, 2, 0) * std + mean       # (H, W, 3)
    img  = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

if __name__ == "__main__":
    print("Running utils.py sanity check…")
 
    # class_ids_to_rgb round-trip
    seg = np.array([[0, 1, 3, 4, 22]], dtype=np.int32)
    rgb = class_ids_to_rgb(seg)
    assert rgb.shape == (1, 5, 3), f"Unexpected shape: {rgb.shape}"
    assert tuple(rgb[0, 1]) == config.CLASS_COLORS[1], "Colour mismatch for class 1"
 
    # depth colourmap
    depth = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    coloured = depth_to_colormap(depth)
    assert coloured.shape == (100, 100, 3)
 
    # bounding box drawing
    dummy_img = np.zeros((600, 800, 3), dtype=np.uint8)
    corners = np.array([[100,100],[200,100],[200,200],[100,200]], dtype=np.float32)
    out = draw_oriented_box(dummy_img, corners)
    assert out.shape == dummy_img.shape
 
 