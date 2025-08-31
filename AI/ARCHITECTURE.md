# AI System Architecture

## Purpose
- Provide multiple AI decision systems (GOAP for adventurers/monsters, community AI for persistent NPCs).

## Adapter contract (game/ai adapter)
- Function signature: take_turn(entity_row, game_state, rng, perception) -> Optional[action]
- action: a dict describing the intended operation. Minimal schema:
  - actor: int (entity id)
  - type: str ("move" | "attack" | "use" | "wait" | "custom")
  - priority: int (optional; higher = processed first)
  - payload: dict of type-specific fields (e.g., {"dx": 1, "dy": 0} for move)
- Handlers that are used in the parallel loop must NOT mutate the authoritative game state directly; they should return:
  - action (or None)
  - updated_ai_state (if internal AI state changed)

## Parallel execution design (planned)
- AI handlers run in parallel over independent chunks, producing (updated_ai_state_chunk, actions_list)
- Main thread merges AI state chunks and applies actions deterministically using a central step that resolves conflicts.

## Action format (reference)
- Example move action:
  ```
  {
    "actor": 123,
    "type": "move",
    "priority": 0,
    "payload": {"dx": 1, "dy": 0}
  }
  ```

- Adapters should validate actions against this schema before returning them.

## Deterministic application
- A single authoritative step will sort actions by (priority desc, actor id asc) and then apply them in order.
- For movement conflicts (multiple actors target same tile), the first-applied action wins. The rest are either canceled or kept as no-op.
- The resolver must be clearly separated from physics/collision rules; it should return a list of applied actions and the updated positions so the game-state applier can perform side-effects (animation, triggers, combat checks) in a single-threaded, deterministic context.

## Notes / Next steps
- Integrate the action schema into the adapter functions (game/ai/*) and add validators.
- Implement deterministic merging and conflict resolution policies in the main loop.
- Expand tests to cover social actions, uses, attack resolution, and multi-step plans.