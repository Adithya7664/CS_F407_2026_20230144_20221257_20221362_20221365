# knowledge_graph.py — 3-layer Spreading Activation Graph for terrain
#                      penalty scoring based on active package traits.
#
# Architecture (fixed, per assignment spec):
#   Layer 1 — Source nodes   : Fragile, Valuable, Biohazard, Heavy
#   Layer 2 — Property nodes : Hard, Wet, Slippery, Dirty, Visible,
#                              Contaminated, Soft, Unstable
#   Layer 3 — Terrain nodes  : Pavement, Grass, Dirt
#
# Rules:
#   • Edges run Layer1 → Layer2 and Layer2 → Layer3 ONLY
#   • All penalty scores derived solely from edge-weight accumulation
#   • NO if/else chains anywhere in the penalty computation
#   • All final scores capped at 1.0

# Reasoning behind each weight:
#
# Fragile → Soft     : 0.9  fragile items need soft landing above all else
# Fragile → Hard     : 0.7  hard surface is very bad for fragile items
# Fragile → Unstable : 0.5  instability risks tipping/breaking
# Fragile → Slippery : 0.4  slippery risks sliding after drop
#
# Valuable → Visible : 0.8  valuable items should land where they're visible
# Valuable → Dirty   : 0.7  dirt/contamination damages valuable items
# Valuable → Wet     : 0.6  moisture can damage valuable goods
# Valuable → Soft    : 0.5  soft landing protects value
# Valuable → Hard    : 0.4  hard surface risks damage
#
# Biohazard → Contaminated : 0.9  biohazard landing must avoid contamination spread
# Biohazard → Wet          : 0.7  wet surfaces spread biohazards faster
# Biohazard → Dirty        : 0.6  dirty terrain worsens biohazard containment
# Biohazard → Visible      : 0.5  visibility helps response teams
# Biohazard → Slippery     : 0.3  minor concern for biohazard
#
# Heavy → Unstable : 0.9  heavy items can sink/destabilise soft ground
# Heavy → Hard     : 0.2  hard surface is actually OK for heavy items
# Heavy → Soft     : 0.8  soft terrain is bad — item sinks/tips
# Heavy → Slippery : 0.5  heavy item sliding is dangerous

SOURCE_TO_PROPERTY_EDGES: dict[str, dict[str, float]] = {
    "Fragile": {
        "Soft":      0.9,
        "Hard":      0.7,
        "Unstable":  0.5,
        "Slippery":  0.4,
    },
    "Valuable": {
        "Visible":   0.8,
        "Dirty":     0.7,
        "Wet":       0.6,
        "Soft":      0.5,
        "Hard":      0.4,
    },
    "Biohazard": {
        "Contaminated": 0.9,
        "Wet":          0.7,
        "Dirty":        0.6,
        "Visible":      0.5,
        "Slippery":     0.3,
    },
    "Heavy": {
        "Unstable":  0.9,
        "Soft":      0.8,
        "Slippery":  0.5,
        "Hard":      0.2,
    },
}

# Reasoning behind each weight:
#
# Pavement terrain properties:
#   Hard        → Pavement : 0.9  pavement is the hardest terrain
#   Wet         → Pavement : 0.7  pavement gets slippery when wet
#   Slippery    → Pavement : 0.8  pavement is most slippery when wet
#   Dirty       → Pavement : 0.3  pavement is relatively easy to keep clean
#   Visible     → Pavement : 0.2  pavement is open and visible
#   Contaminated→ Pavement : 0.5  contamination spreads on hard surfaces
#   Soft        → Pavement : 0.0  pavement is not soft at all
#   Unstable    → Pavement : 0.1  pavement is generally stable
#
# Grass terrain properties:
#   Hard        → Grass : 0.1  grass is not hard
#   Wet         → Grass : 0.5  grass retains moisture
#   Slippery    → Grass : 0.4  wet grass is slippery
#   Dirty       → Grass : 0.5  grass can be muddy/dirty
#   Visible     → Grass : 0.4  grass visibility is moderate
#   Contaminated→ Grass : 0.6  grass absorbs contaminants
#   Soft        → Grass : 0.1  grass IS soft — LOW penalty for soft trait
#   Unstable    → Grass : 0.4  grass can be uneven
#
# Dirt terrain properties:
#   Hard        → Dirt : 0.3  dirt is semi-hard depending on dryness
#   Wet         → Dirt : 0.8  wet dirt becomes mud — very problematic
#   Slippery    → Dirt : 0.6  muddy dirt is slippery
#   Dirty       → Dirt : 0.9  dirt IS dirty by definition
#   Visible     → Dirt : 0.5  dirt has moderate visibility
#   Contaminated→ Dirt : 0.8  dirt absorbs and spreads contamination
#   Soft        → Dirt : 0.5  dirt is moderately soft
#   Unstable    → Dirt : 0.7  loose dirt is unstable

PROPERTY_TO_TERRAIN_EDGES: dict[str, dict[str, float]] = {
    "Hard": {
        "Pavement": 0.9,
        "Grass":    0.1,
        "Dirt":     0.3,
    },
    "Wet": {
        "Pavement": 0.7,
        "Grass":    0.5,
        "Dirt":     0.8,
    },
    "Slippery": {
        "Pavement": 0.8,
        "Grass":    0.4,
        "Dirt":     0.6,
    },
    "Dirty": {
        "Pavement": 0.3,
        "Grass":    0.5,
        "Dirt":     0.9,
    },
    "Visible": {
        "Pavement": 0.2,
        "Grass":    0.4,
        "Dirt":     0.5,
    },
    "Contaminated": {
        "Pavement": 0.5,
        "Grass":    0.6,
        "Dirt":     0.8,
    },
    "Soft": {
        "Pavement": 0.0,
        "Grass":    0.1,   # grass is soft → LOW penalty when Soft trait active
        "Dirt":     0.5,
    },
    "Unstable": {
        "Pavement": 0.1,
        "Grass":    0.4,
        "Dirt":     0.7,
    },
}
 
# All valid terrain node names
TERRAIN_NODES = ["Pavement", "Grass", "Dirt"]
 
# All valid source node names
SOURCE_NODES = list(SOURCE_TO_PROPERTY_EDGES.keys())

#spreading
def think(active_nodes: list[str]) -> dict[str, float]:
    """
    Run a 2-hop spreading activation from active source nodes to terrain nodes.
 
    Step 1: For each active source node, propagate its unit activation (1.0)
            along SOURCE_TO_PROPERTY_EDGES, accumulating weighted signals at
            each property node.
 
    Step 2: Propagate accumulated property signals along
            PROPERTY_TO_TERRAIN_EDGES, accumulating weighted signals at each
            terrain node.
 
    Step 3: Cap all terrain scores at 1.0.
 
    Parameters
    ----------
    active_nodes : list[str]
        Names of active Layer-1 source nodes, e.g. ["Fragile", "Valuable"].
        Names not in SOURCE_NODES are silently ignored.
 
    Returns
    -------
    terrain_scores : dict[str, float]
        Penalty score for each terrain type, e.g.:
        {"Pavement": 0.84, "Grass": 0.27, "Dirt": 0.61}
        Higher score = worse terrain for this package.
    """
    #source to property
    property_activation: dict[str, float] = {}
 
    for source in active_nodes:
        edges = SOURCE_TO_PROPERTY_EDGES.get(source, {})
        for prop, weight in edges.items():
            property_activation[prop] = (
                property_activation.get(prop, 0.0) + weight
            )
    #property to terrain
    terrain_activation: dict[str, float] = {t: 0.0 for t in TERRAIN_NODES}
 
    for prop, p_signal in property_activation.items():
        terrain_edges = PROPERTY_TO_TERRAIN_EDGES.get(prop, {})
        for terrain, weight in terrain_edges.items():
            terrain_activation[terrain] = (
                terrain_activation[terrain] + p_signal * weight
            )
            
    terrain_scores = {
        terrain: min(score, 1.0)
        for terrain, score in terrain_activation.items()
    }
 
    return terrain_scores

#terrain name-class mapping
TERRAIN_TO_CLASS_IDS: dict[str, list[int]] = {
    "Pavement": [1],
    "Grass":    [3],
    "Dirt":     [4],
}
 
CLASS_ID_TO_TERRAIN: dict[int, str] = {
    cls_id: terrain
    for terrain, ids in TERRAIN_TO_CLASS_IDS.items()
    for cls_id in ids
}
 
 
def get_terrain_penalty_map(active_nodes: list[str]) -> dict[int, float]:
    """
    Convenience wrapper: returns a dict keyed by class ID (int) rather
    than terrain name (str), for direct use in geometry.compute_costs().
 
    Parameters
    ----------
    active_nodes : list[str]  — active source node names
 
    Returns
    -------
    {class_id: penalty_score}  e.g. {1: 0.84, 3: 0.27, 4: 0.61}
    """
    terrain_scores = think(active_nodes)
    return {
        cls_id: terrain_scores.get(terrain, 0.0)
        for cls_id, terrain in CLASS_ID_TO_TERRAIN.items()
    }
    
if __name__ == "__main__":
    print("Running knowledge_graph.py sanity check…\n")
 
    # Test 1: Fragile + Valuable → grass should be lowest penalty
    scores = think(["Fragile", "Valuable"])
    print("Active nodes: [Fragile, Valuable]")
    for t, s in scores.items():
        print(f"  {t:<12}: {s:.4f}")
    assert scores["Grass"] < scores["Pavement"], \
        "Grass should have lower penalty than Pavement for Fragile+Valuable"
    assert scores["Grass"] < scores["Dirt"], \
        "Grass should have lower penalty than Dirt for Fragile+Valuable"
    print("  ✓ Grass is lowest (soft landing preferred)\n")
 
    # Test 2: Heavy → grass/dirt should be penalised (soft/unstable)
    scores2 = think(["Heavy"])
    print("Active nodes: [Heavy]")
    for t, s in scores2.items():
        print(f"  {t:<12}: {s:.4f}")
    assert scores2["Grass"] > scores2["Pavement"], \
        "Pavement should be better than Grass for Heavy packages"
    print("  ✓ Pavement preferred for heavy packages\n")
 
    # Test 3: Biohazard → dirt should be worst (absorbs/spreads contamination)
    scores3 = think(["Biohazard"])
    print("Active nodes: [Biohazard]")
    for t, s in scores3.items():
        print(f"  {t:<12}: {s:.4f}")
    assert scores3["Dirt"] > scores3["Pavement"], \
        "Dirt should be worse than Pavement for Biohazard"
    print("  ✓ Dirt most penalised for biohazard\n")
 
    # Test 4: No active nodes → all zeros
    scores4 = think([])
    print("Active nodes: []")
    for t, s in scores4.items():
        print(f"  {t:<12}: {s:.4f}")
    assert all(s == 0.0 for s in scores4.values()), \
        "All scores should be 0 when no nodes active"
    print("  ✓ All zeros when no active nodes\n")
 
    # Test 5: class-ID penalty map
    penalty_map = get_terrain_penalty_map(["Fragile", "Valuable"])
    print(f"Class-ID penalty map: {penalty_map}")