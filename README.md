# BasicRL

**BasicRL** is an experimental Python-based roguelike engine focused on deep systemic simulation and high performance. It aims to serve both as the foundation for a feature-rich single‑player game and as a sandbox for experimenting with heuristic and machine‑learned AI. The engine's content is driven by internal configuration files; external mods are unsupported.

> **Note:** This repository is private and experimental. It is not open for external use or contributions.

## Features

- **Engine Core** – PySide6 window manager, tile renderer, and main loop built for deterministic turn-based simulation.
- **Data-Driven World** – Game state, entities, items, and effects stored in Polars DataFrames for fast, schema‑validated access.
- **Procedural Generation** – BSP dungeon generator producing rooms, corridors, and basic height data.
- **Entity & Item Systems** – Body-plan driven equipment slots, attachable items, resources like fuel and durability, and effect execution.
- **Lighting & FOV** – Numba-accelerated shadowcasting with optional colored lights and height visualization.
- **Internal Configuration** – YAML/TOML files define tilesets, items, entities, keybindings, and rules for in-house development. External modding is unsupported.
- **Future AI/ML Integration** – GOAP-style adapters with a planned AI dispatcher to allow swapping in heuristic or machine-learned policies per entity.

## Roadmap

Planned milestones include:

- Expanded dungeon generation (caves, multi-level maps, above-ground prefabs).
- Proximity-based world simulation tiers.
- Combat, stealth, magic, and saving/loading systems.
- Internal tooling and documentation.

See `basicrl_project.txt` for a detailed task list and vision statement.

## Installation

BasicRL targets **Python 3.11**. Runtime dependencies are pinned in `requirements.txt` and development tools live in `requirements-dev.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

Alternatively, an `environment.yml` is provided for Conda users.

## Running the game

After installing dependencies, launch the prototype game loop via the wrapper script:

```bash
./run.sh
```

The window will display a procedurally generated dungeon where you can move the @‑like player, pick up items, and view a basic inventory.

## Testing

Unit tests cover core systems such as pathfinding, FOV, perception, and inventory handling. Run them with:

```bash
pytest
```

Some tests currently require optional dependencies like NumPy, Polars, and structlog.

## Contributing

 
This project is private and not open to external contributions or use at this time.

## License

This project's license is pending.

---

BasicRL is a private, experimental work in progress. External contributions or use are not permitted.
