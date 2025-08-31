# BasicRL

**BasicRL** is an experimental Python-based roguelike engine focused on deep systemic simulation, high performance, and modding. It aims to serve both as the foundation for a feature-rich single‑player game and as a sandbox for experimenting with heuristic and machine‑learned AI.

## Features

- **Engine Core** – PySide6 window manager, tile renderer, and main loop built for deterministic turn-based simulation.
- **Data-Driven World** – Game state, entities, items, and effects stored in Polars DataFrames for fast, schema‑validated access.
- **Procedural Generation** – BSP dungeon generator producing rooms, corridors, and basic height data.
- **Entity & Item Systems** – Body-plan driven equipment slots, attachable items, resources like fuel and durability, and effect execution.
- **Lighting & FOV** – Numba-accelerated shadowcasting with optional colored lights and height visualization.
- **Modding Friendly** – YAML/TOML configuration for tilesets, items, entities, keybindings, and rules. Designed to let modders extend content without touching engine code.
- **Future AI/ML Integration** – GOAP-style adapters with a planned AI dispatcher to allow swapping in heuristic or machine-learned policies per entity.

## Roadmap

Planned milestones include:

- Expanded dungeon generation (caves, multi-level maps, above-ground prefabs).
- Proximity-based world simulation tiers.
- Combat, stealth, magic, and saving/loading systems.
- Modding guides and contributor documentation.

See `basicrl_project.txt` for a detailed task list and vision statement.

## Installation

BasicRL targets **Python 3.11**. Install dependencies using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Alternatively, an `environment.yml` is provided for Conda users.

## Running the game

After installing dependencies, launch the prototype game loop:

```bash
python main.py
```

The window will display a procedurally generated dungeon where you can move the @‑like player, pick up items, and view a basic inventory.

## Testing

Unit tests cover core systems such as pathfinding, FOV, perception, and inventory handling. Run them with:

```bash
pytest
```

Some tests currently require optional dependencies like NumPy, Polars, and structlog.

## Contributing

Pull requests are welcome! Please ensure new code is well‑documented and covered by tests when possible.

1. Fork and clone the repository.
2. Create a Python 3.11 virtual environment and install dependencies.
3. Make your changes on a feature branch.
4. Run `pytest` and add/update documentation.
5. Submit a pull request describing your changes and testing steps.

## License

This project's license is pending.

---

BasicRL is a work in progress. Feedback, ideas, and contributions are encouraged.
