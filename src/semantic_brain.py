import json
import os
import numpy as np
 
import config
from knowledge_graph import (
    CLASS_ID_TO_TERRAIN,
    TERRAIN_NODES,
    SOURCE_NODES,
    think,
    get_terrain_penalty_map,
)
#Terrain classification
def get_dominant_terrain(seg_map: np.ndarray,
                         pts: np.ndarray) -> tuple[str, int]:
    """
    Identify the dominant terrain type within a candidate bounding box
    by sampling its interior points on the segmentation map.
 
    Only safe classes (config.SAFE_CLASSES = [1, 3, 4]) are considered.
    If no safe-class pixels are found, falls back to the overall mode class.
 
    Parameters
    ----------
    seg_map : np.ndarray  shape (H, W),  dtype int  — class IDs at 800×600
    pts     : np.ndarray  shape (N, 2),  dtype float32 — (x,y) sample points
              produced by geometry._interior_sample_points()
 
    Returns
    -------
    terrain_name : str   — e.g. "Grass", "Pavement", "Dirt"
    class_id     : int   — the dominant safe class ID (1, 3, or 4)
    """
    H, W = seg_map.shape
    class_counts: dict[int, int] = {}
 
    for x, y in pts:
        xi, yi = int(round(float(x))), int(round(float(y)))
        if 0 <= xi < W and 0 <= yi < H:
            cls = int(seg_map[yi, xi])
            class_counts[cls] = class_counts.get(cls, 0) + 1
 
    if not class_counts:
        return "Pavement", 1      # safe default
 
    # Prefer safe classes; fall back to overall mode if none present
    safe_counts = {
        cls: cnt for cls, cnt in class_counts.items()
        if cls in config.SAFE_CLASSES
    }
 
    counts_to_use = safe_counts if safe_counts else class_counts
    dominant_id   = max(counts_to_use, key=counts_to_use.get)
 
    terrain_name  = CLASS_ID_TO_TERRAIN.get(dominant_id, "Pavement")
    return terrain_name, dominant_id
 
 
def classify_all_candidates(candidates: list[dict],
                             seg_map:    np.ndarray) -> list[dict]:
    """
    Attach terrain_name and terrain_class_id to every candidate dict
    in-place (and returns the same list for chaining).
 
    Parameters
    ----------
    candidates : list of dicts from geometry.find_candidate_boxes()
    seg_map    : (H, W) int class-ID map at 800×600
 
    Returns
    -------
    candidates : same list, each dict now has:
        terrain_name     : str   dominant terrain type
        terrain_class_id : int   dominant class ID
    """
    for cand in candidates:
        name, cls_id = get_dominant_terrain(seg_map, cand["pts"])
        cand["terrain_name"]     = name
        cand["terrain_class_id"] = cls_id
    return candidates

#mission config parser
# Maps JSON property key → Layer-1 source node name in the knowledge graph
_TRAIT_TO_SOURCE_NODE: dict[str, str] = {
    "fragile":  "Fragile",
    "valuable": "Valuable",
    "biohazard":"Biohazard",
    "heavy":    "Heavy",
}
 
 
def parse_mission_config(path: str = config.MISSION_CONFIG_PATH) -> list[str]:
    """
    Read mission_config.json and extract the list of active source nodes
    for the knowledge graph.
 
    Expected JSON structure:
    {
        "mission_id": "OP-DELTA-9",
        "package": {
            "type":      "medical_vials",
            "heavy":     true,
            "fragile":   false,
            "valuable":  false,
            "biohazard": false
        }
    }
 
    A trait activates its source node if and only if its value is true.
 
    Parameters
    ----------
    path : str  — path to mission_config.json
 
    Returns
    -------
    active_nodes : list[str]
        e.g. ["Heavy"] for the example above.
        Returns [] if the file is missing or the package block is absent.
    """
    if not os.path.isfile(path):
        print(f"[semantic_brain] Warning: '{path}' not found. "
              f"Using no active nodes.")
        return []
 
    with open(path, "r", encoding="utf-8") as f:
        mission = json.load(f)
 
    package = mission.get("package", {})
    if not package:
        print("[semantic_brain] Warning: 'package' block missing in config.")
        return []
 
    active_nodes = [
        source_node
        for trait, source_node in _TRAIT_TO_SOURCE_NODE.items()
        if package.get(trait, False) is True
    ]
 
    print(f"[semantic_brain] Mission ID   : {mission.get('mission_id', 'N/A')}")
    print(f"[semantic_brain] Package type : {package.get('type', 'unknown')}")
    print(f"[semantic_brain] Active nodes : {active_nodes}")
 
    return active_nodes

#semantic scoring pipeline
def compute_semantic_scores(mission_config_path: str = config.MISSION_CONFIG_PATH
                             ) -> tuple[list[str], dict[int, float]]:
    """
    End-to-end helper that:
      1. Parses mission_config.json → active source nodes
      2. Runs the knowledge graph → terrain penalty scores
      3. Returns both for use in geometry.compute_costs()
 
    Parameters
    ----------
    mission_config_path : str  — path to mission_config.json
 
    Returns
    -------
    active_nodes    : list[str]          e.g. ["Fragile", "Valuable"]
    penalty_by_id   : dict[int, float]   e.g. {1: 0.84, 3: 0.10, 4: 0.62}
    """
    active_nodes  = parse_mission_config(mission_config_path)
    penalty_by_id = get_terrain_penalty_map(active_nodes)
 
    print("\n[semantic_brain] Terrain penalty scores (from knowledge graph):")
    for cls_id, penalty in penalty_by_id.items():
        terrain = CLASS_ID_TO_TERRAIN.get(cls_id, "?")
        print(f"  Class {cls_id} ({terrain:<10}): {penalty:.4f}")
 
    return active_nodes, penalty_by_id
 
 
def get_penalty_for_candidate(candidate: dict,
                               active_nodes: list[str]) -> float:
    """
    Retrieve the semantic penalty score for a single candidate using
    its already-classified terrain name.
 
    Parameters
    ----------
    candidate    : dict with key "terrain_name" set by classify_all_candidates()
    active_nodes : list of active source node names
 
    Returns
    -------
    penalty : float  in [0.0, 1.0]
    """
    terrain_scores = think(active_nodes)
    terrain_name   = candidate.get("terrain_name", "Pavement")
    return terrain_scores.get(terrain_name, 0.0)

if __name__ == "__main__":
    print("Running semantic_brain.py sanity check…\n")
    #terrain classification
    seg = np.ones((config.MAIN_H, config.MAIN_W), dtype=np.int32)
    seg[200:400, 200:600] = 3    # grass patch
    seg[400:500, 200:600] = 4    # dirt patch
 
    pts_on_grass = np.array([[300.0, 300.0],
                              [350.0, 350.0],
                              [400.0, 250.0]], dtype=np.float32)
    name, cls_id = get_dominant_terrain(seg, pts_on_grass)
    print(f"Dominant terrain (should be Grass): {name}  class_id={cls_id}")
    assert name == "Grass" and cls_id == 3, "Terrain classification failed!"
    print("  ✓ Grass detected correctly\n")
 
    pts_on_dirt = np.array([[300.0, 450.0],
                             [350.0, 420.0]], dtype=np.float32)
    name2, cls2 = get_dominant_terrain(seg, pts_on_dirt)
    print(f"Dominant terrain (should be Dirt) : {name2}  class_id={cls2}")
    assert name2 == "Dirt" and cls2 == 4, "Terrain classification failed!"
    print("  ✓ Dirt detected correctly\n")
    
    #mission config parser
    import tempfile, json as _json
 
    sample_config = {
        "mission_id": "TEST-01",
        "package": {
            "type":      "test_package",
            "heavy":     True,
            "fragile":   True,
            "valuable":  False,
            "biohazard": False,
        }
    }
 
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        _json.dump(sample_config, tmp)
        tmp_path = tmp.name
 
    active = parse_mission_config(tmp_path)
    print(f"\nParsed active nodes: {active}")
    assert "Heavy"   in active, "Heavy should be active"
    assert "Fragile" in active, "Fragile should be active"
    assert "Valuable" not in active, "Valuable should NOT be active"
    print("  ✓ Mission config parsed correctly\n")
 
    os.unlink(tmp_path)
    
    #full piepeline
    print("Full pipeline check (using sample config)…")
    sample_config2 = {
        "mission_id": "OP-DELTA-9",
        "package": {
            "type": "medical_vials",
            "heavy": True,
            "fragile": False,
            "valuable": False,
            "biohazard": False,
        }
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp2:
        _json.dump(sample_config2, tmp2)
        tmp2_path = tmp2.name
 
    nodes, penalties = compute_semantic_scores(tmp2_path)
    os.unlink(tmp2_path)
 
    print(f"  Active nodes  : {nodes}")
    print(f"  Penalty map   : {penalties}")
    assert isinstance(penalties, dict)
    assert all(0.0 <= v <= 1.0 for v in penalties.values()), \
        "All penalties must be in [0, 1]"