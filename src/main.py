# main.py — Entry point and orchestration layer for the drone landing zone
#           AI system (CS F407 Project Assignment II).
#
# Pipeline (in order):
#   1. CLI input  : image path + target (x, y) coordinate
#   2. Load models: SegNet (best_model.pth) + MiDaS_small (torch.hub)
#   3. Segmentation  : image → 256×256 ANN → upscale mask → 800×600
#   4. Depth          : image → MiDaS_small → normalised uint8 depth map
#   5. Safe mask      : seg map → binary white/black mask
#   6. Candidate search: rotational grid search over safe mask
#   7. Terrain classify: dominant class per candidate bounding box
#   8. Semantic graph  : mission_config.json → active nodes → penalties
#   9. Cost ranking    : distance + roughness + semantic → sort candidates
#  10. Render output   : output.jpg  +  output_analysis.jpg
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse

import cv2
import numpy as np
import torch
from torchvision import transforms

import config
import utils
import geometry
import semantic_brain
from train import load_model


# ── 1. MiDaS depth model loader ───────────────────────────────────────────────

def load_midas(device: torch.device):
    """
    Load MiDaS_small from torch.hub.

    Returns
    -------
    midas       : nn.Module  (eval mode, on device)
    transform   : callable   MiDaS preprocessing transform
    """
    print("[main] Loading MiDaS_small via torch.hub…")
    midas = torch.hub.load(
        config.MIDAS_REPO,
        config.MIDAS_MODEL,
        pretrained=True,
        trust_repo=True,
    )
    midas.to(device).eval()

    transforms_hub = torch.hub.load(
        config.MIDAS_REPO,
        "transforms",
        trust_repo=True,
    )
    transform = transforms_hub.small_transform
    print("[main] MiDaS loaded ✓")
    return midas, transform


# ── 2. Segmentation inference ─────────────────────────────────────────────────

def run_segmentation(model, image_800: np.ndarray,
                     device: torch.device) -> np.ndarray:
    """
    Run the trained SegNet on a single 800×600 RGB image.

    Steps:
      • Downscale to 256×256 + normalise  → feed ANN
      • Argmax logits → (256, 256) class-ID map
      • Upscale back to 800×600 via nearest-neighbour

    Parameters
    ----------
    model     : SegNet  (eval mode)
    image_800 : (600, 800, 3) uint8 RGB
    device    : torch.device

    Returns
    -------
    seg_map : (600, 800) int32  class-ID map at 800×600
    """
    # Preprocess: resize to 256×256 and normalise
    ann_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.ANN_H, config.ANN_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    tensor = ann_transform(image_800).unsqueeze(0).to(device)  # (1,3,256,256)

    with torch.no_grad():
        logits = model(tensor)                    # (1, 23, 256, 256)

    # Argmax → (256, 256) class IDs
    seg_256 = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

    # Upscale to 800×600 (nearest-neighbour preserves class labels)
    seg_800 = geometry.upscale_seg_map(seg_256)   # (600, 800) int32

    return seg_800


# ── 3. MiDaS depth inference ──────────────────────────────────────────────────

def run_depth(midas, transform, image_800: np.ndarray,
              device: torch.device) -> np.ndarray:
    """
    Run MiDaS_small on the 800×600 RGB image to produce a normalised
    uint8 depth map at the same resolution.

    Bright pixels = close to drone (high value).
    Dark  pixels  = far from drone (low value).

    Parameters
    ----------
    midas     : MiDaS model (eval mode)
    transform : MiDaS preprocessing transform
    image_800 : (600, 800, 3) uint8 RGB

    Returns
    -------
    depth_uint8 : (600, 800) uint8  normalised depth map [0–255]
    """
    # MiDaS expects BGR (OpenCV convention)
    bgr = cv2.cvtColor(image_800, cv2.COLOR_RGB2BGR)
    input_tensor = transform(bgr).to(device)      # MiDaS handles its own sizing

    with torch.no_grad():
        prediction = midas(input_tensor)

    # Resize prediction back to 800×600
    pred_np = prediction.squeeze().cpu().numpy()
    depth_resized = cv2.resize(pred_np,
                               (config.MAIN_W, config.MAIN_H),
                               interpolation=cv2.INTER_LINEAR)

    # Normalise to uint8 [0–255]  (higher value = closer to drone)
    d_min, d_max = depth_resized.min(), depth_resized.max()
    if d_max - d_min > 1e-6:
        depth_norm = (depth_resized - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_resized)

    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    return depth_uint8


# ── 4. Full pipeline ───────────────────────────────────────────────────────────

def run_pipeline(image_path: str,
                 target_x:   int,
                 target_y:   int,
                 seg_model,
                 midas,
                 midas_transform,
                 device:     torch.device):
    """
    Execute the complete inference pipeline for one image and target point.

    Parameters
    ----------
    image_path      : path to the input drone image
    target_x/y      : user-specified target pixel in 800×600 space
    seg_model       : loaded SegNet (eval mode)
    midas           : loaded MiDaS (eval mode)
    midas_transform : MiDaS preprocessing transform
    device          : torch.device

    Side-effects
    ------------
    Saves output.jpg and output_analysis.jpg to the working directory.
    """
    print(f"\n{'='*60}")
    print(f"  Image  : {image_path}")
    print(f"  Target : ({target_x}, {target_y})")
    print(f"{'='*60}\n")

    # ── Step 1: Load + resize to 800×600 ─────────────────────────────────────
    print("[main] Loading image…")
    raw = utils.load_image_rgb(image_path)
    image_800 = utils.resize_to_main(raw)           # (600, 800, 3) uint8 RGB
    print(f"       Image resized to {image_800.shape[1]}×{image_800.shape[0]}")

    # ── Step 2: Semantic segmentation ────────────────────────────────────────
    print("[main] Running segmentation (SegNet)…")
    seg_map = run_segmentation(seg_model, image_800, device)
    seg_rgb = utils.class_ids_to_rgb(seg_map)       # (600, 800, 3) for display
    print(f"       Unique classes found: {np.unique(seg_map).tolist()}")

    # ── Step 3: Depth estimation ──────────────────────────────────────────────
    print("[main] Running depth estimation (MiDaS)…")
    depth_map  = run_depth(midas, midas_transform, image_800, device)
    depth_rgb  = utils.depth_to_colormap(depth_map)
    print(f"       Depth range: [{depth_map.min()}, {depth_map.max()}]")

    # ── Step 4: Binary safe mask ──────────────────────────────────────────────
    print("[main] Building binary safe mask…")
    safe_mask  = geometry.build_safe_mask(seg_map)
    safe_rgb   = utils.binary_mask_to_rgb(safe_mask)
    n_safe     = int((safe_mask == 255).sum())
    print(f"       Safe pixels: {n_safe} / {safe_mask.size} "
          f"({100*n_safe/safe_mask.size:.1f}%)")

    # ── Step 5: Candidate search ──────────────────────────────────────────────
    print("[main] Searching for candidate landing zones…")
    candidates = geometry.find_candidate_boxes(safe_mask, depth_map)
    print(f"       Candidates found: {len(candidates)}")

    if not candidates:
        print("\n[main] ✗ No valid landing zone found in this image.")
        print("       Try a different target coordinate or image.")
        sys.exit(1)

    # ── Step 6: Terrain classification ───────────────────────────────────────
    print("[main] Classifying terrain per candidate…")
    candidates = semantic_brain.classify_all_candidates(candidates, seg_map)

    # ── Step 7: Semantic scoring via knowledge graph ──────────────────────────
    print("[main] Running knowledge graph (semantic_brain)…")
    active_nodes, penalty_by_id = semantic_brain.compute_semantic_scores(
        config.MISSION_CONFIG_PATH
    )

    # ── Step 8: Cost ranking ──────────────────────────────────────────────────
    print("[main] Computing and ranking costs…")
    ranked = geometry.compute_costs(
        candidates,
        target_xy=(float(target_x), float(target_y)),
        semantic_scores=penalty_by_id,
        seg_map=seg_map,
    )

    best = ranked[0]
    print(f"\n[main] ✓ Best landing zone found:")
    print(f"       Position : ({int(best['cx'])}, {int(best['cy'])})")
    print(f"       Angle    : {best['angle']}°")
    print(f"       Terrain  : {best.get('terrain_name', '?')}")
    print(f"       Distance : {best['distance']:.4f}")
    print(f"       Roughness: {best['roughness']:.4f}")
    print(f"       Semantic : {best['semantic_penalty']:.4f}")
    print(f"       TOTAL    : {best['cost']:.4f}")

    # ── Step 9: Render output.jpg ─────────────────────────────────────────────
    print("\n[main] Rendering output images…")

    # Draw target marker
    result = utils.draw_target_marker(image_800,
                                      xy=(target_x, target_y),
                                      color=(255, 0, 0))

    # Draw best bounding box (green)
    result = utils.draw_oriented_box(result,
                                     best["corners"],
                                     color=(0, 255, 0),
                                     thickness=2)

    # Overlay cost label
    result = utils.draw_cost_label(result,
                                   cx=int(best["cx"]),
                                   cy=int(best["cy"]),
                                   cost=best["cost"],
                                   terrain=best.get("terrain_name", ""),
                                   color=(0, 255, 0))

    utils.save_image_rgb(config.OUTPUT_IMAGE, result)

    # ── Step 10: Render output_analysis.jpg (6-panel dashboard) ──────────────
    # Build placement panel (same as result but smaller context)
    placement_panel = result.copy()

    # Build cost text panel
    cost_panel = utils.build_cost_text_panel(best, active_nodes)

    # Update "Best Placement" panel title with actual cost
    dashboard = utils.build_analysis_dashboard(
        original=image_800,
        seg_rgb=seg_rgb,
        depth_rgb=depth_rgb,
        safe_rgb=safe_rgb,
        placement=placement_panel,
        cost_panel=cost_panel,
    )

    utils.save_image_rgb(config.OUTPUT_ANALYSIS, dashboard)

    print(f"\n[main] Done.")
    print(f"       {config.OUTPUT_IMAGE}    — image with best bounding box")
    print(f"       {config.OUTPUT_ANALYSIS} — 6-panel analysis dashboard")

    return best, active_nodes


# ── 5. CLI entry point ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "CS F407 — Drone Landing Zone AI\n"
            "Finds the safest + closest package drop zone in a drone image.\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to the input drone image (any resolution).",
    )
    parser.add_argument(
        "--target-x", "-x",
        type=int,
        default=None,
        help="Target X pixel coordinate in 800×600 space (0 ≤ x ≤ 800).",
    )
    parser.add_argument(
        "--target-y", "-y",
        type=int,
        default=None,
        help="Target Y pixel coordinate in 800×600 space (0 ≤ y ≤ 600).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=config.CHECKPOINT_PATH,
        help=f"Path to SegNet checkpoint (default: {config.CHECKPOINT_PATH}).",
    )
    parser.add_argument(
        "--mission-config",
        type=str,
        default=config.MISSION_CONFIG_PATH,
        help=f"Path to mission_config.json (default: {config.MISSION_CONFIG_PATH}).",
    )
    return parser


def interactive_prompt(parser) -> argparse.Namespace:
    """
    If CLI args are missing, prompt the user interactively.
    Matches the assignment's required interactive CLI behaviour.
    """
    args = parser.parse_args()

    if args.image is None:
        print("\n" + "="*60)
        print("  CS F407 — Drone Landing Zone AI")
        print("="*60)
        args.image = input("\nEnter image file path: ").strip()

    if not os.path.isfile(args.image):
        print(f"[main] Error: image file '{args.image}' not found.")
        sys.exit(1)

    if args.target_x is None:
        while True:
            try:
                args.target_x = int(input("Enter target X coordinate (0–800): ").strip())
                if 0 <= args.target_x <= config.MAIN_W:
                    break
                print(f"  X must be between 0 and {config.MAIN_W}.")
            except ValueError:
                print("  Please enter a valid integer.")

    if args.target_y is None:
        while True:
            try:
                args.target_y = int(input("Enter target Y coordinate (0–600): ").strip())
                if 0 <= args.target_y <= config.MAIN_H:
                    break
                print(f"  Y must be between 0 and {config.MAIN_H}.")
            except ValueError:
                print("  Please enter a valid integer.")

    return args


def main():
    parser   = parse_args()
    args     = interactive_prompt(parser)
    device   = config.DEVICE

    # Update mission config path if overridden via CLI
    if args.mission_config != config.MISSION_CONFIG_PATH:
        config.MISSION_CONFIG_PATH = args.mission_config

    # ── Load models ────────────────────────────────────────────────────────────
    print(f"\n[main] Device: {device}")

    print("[main] Loading SegNet from checkpoint…")
    seg_model = load_model(checkpoint=args.checkpoint, device=device)

    midas, midas_transform = load_midas(device)

    # ── Run pipeline ───────────────────────────────────────────────────────────
    run_pipeline(
        image_path=args.image,
        target_x=args.target_x,
        target_y=args.target_y,
        seg_model=seg_model,
        midas=midas,
        midas_transform=midas_transform,
        device=device,
    )


if __name__ == "__main__":
    main()
    