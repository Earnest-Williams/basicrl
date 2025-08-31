"""Deterministic, minimal action resolver for merging AI-produced actions.

This module provides a simple apply_actions function. It operates on lightweight data
structures to avoid coupling to the full game state. It is intended as a reference
implementation to guide the design of the authoritative action-application step.

apply_actions(actions, positions, walkable) -> (new_positions, applied_actions)
- actions: list of dicts matching the action_schema (must include 'actor', 'type', 'payload')
- positions: dict mapping actor_id -> (x, y)
- walkable: set of (x,y) tuples that are considered walkable

Resolution policy (simple):
- Sort by (-priority, actor_id)
- For move actions, compute desired target = (x+dx, y+dy)
- If target not in walkable, skip action
- If multiple actors target the same tile, the first sorted actor wins and is moved; others are skipped
"""
from typing import Dict, List, Tuple, Any


def apply_actions(actions: List[Dict[str, Any]], positions: Dict[int, Tuple[int, int]], walkable: set):
    # Sort deterministically: higher priority first, then lower actor id
    sorted_actions = sorted(actions, key=lambda a: (-int(a.get("priority", 0)), int(a["actor"])))
    desired_targets = {}
    new_positions = positions.copy()
    occupied_targets = set(positions.values())
    applied = []

    for act in sorted_actions:
        actor = act["actor"]
        typ = act.get("type")
        payload = act.get("payload", {})
        if typ == "move":
            if actor not in new_positions:
                # Unknown actor, skip
                continue
            x, y = new_positions[actor]
            dx = int(payload.get("dx", 0))
            dy = int(payload.get("dy", 0))
            target = (x + dx, y + dy)
            # Validate walkable
            if target not in walkable:
                # Skip invalid moves
                continue
            # Check if someone already claimed this target
            if target in desired_targets:
                # Someone with higher priority/earlier order already claimed it; skip
                continue
            # Otherwise claim and apply
            desired_targets[target] = actor
            # Update position (we don't check for swaps in this minimal impl)
            new_positions[actor] = target
            applied.append(act)
        else:
            # For non-move actions, accept them for now (they may be handled later)
            applied.append(act)

    return new_positions, applied