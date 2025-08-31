# Modding Guide

BasicRL is designed to load most game content from data files so that new content can be added without touching engine code.  This guide covers how to extend the game by editing configuration files and adding Python modules.

## Configuration driven assets

All core assets live in the `config/` directory and are written in YAML or TOML:

- `config.yaml` – global engine options
- `entities.yaml` – templates for monsters, NPCs and portals
- `items.yaml` – weapon and armor templates
- `effects.yaml` – reusable effect definitions
- `keybindings.toml`, `overlays.toml`, `settings.toml` – user interface and control configuration

The engine loads these files on start, so changes are picked up the next time the game is launched.

## Adding entities

1. Open `config/entities.yaml`.
2. Under the `templates:` section, add a new key for the entity (for example `goblin_chief`).
3. Provide fields such as `glyph`, `color`, `hp`, and `name` mirroring existing examples.
4. Save the file and launch the game or run tests to verify the new entity can be spawned.

## Adding items

1. Edit `config/items.yaml`.
2. Create a new entry inside `templates:` with `name`, `glyph`, `item_type`, and optional `attributes` and `effects`.
3. Use the `flags` list to mark special behavior like `EQUIPPABLE` or `MAGICAL`.  Combat-related flags such as `TWO_HANDED` or `OFF_HAND` influence how weapons are treated during damage calculation.
4. Save the file and load the game to see the new item.

## Adding effects

1. Add a new effect definition to `config/effects.yaml` under `effects:`.
2. Set the `type` (`active` or `triggered`), `target_category`, and `logic_handler`.
3. The `logic_handler` field references a Python function in `game/effects/handlers.py`.
4. Provide any parameters in `params:`; these are passed to the handler at runtime.

## Plugging in new AI

AI behaviours live under the `AI/` folder.  Create a new module that implements the desired behaviour and expose a class or function entry point.  Entity templates can reference this behaviour via fields in `entities.yaml` or by custom game code that selects an AI implementation per entity.

## Custom generation algorithms

Dungeon generation is managed by the modules in the `Dungeon/` directory.  To introduce a new generation algorithm, add a Python module implementing the generator and update the launch code (e.g. `main.py`) to call it instead of the default pipeline.  Generators should return a data structure compatible with the engine's map loader so that entities and items can be placed on the resulting map.

## Testing mods

After editing configuration or adding code, run the game's unit tests to ensure nothing regresses:

```bash
pytest
```

Testing early helps catch schema mistakes or Python errors before launching the game.
