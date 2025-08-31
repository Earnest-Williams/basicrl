# BasicRL - Python Roguelike Game Engine

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

- **Install dependencies first**:
  - `pip install polars numpy numba pyside6 pillow pyyaml structlog scipy scikit-image joblib pytest`
  - NEVER CANCEL: Dependency installation takes 3-5 minutes. Set timeout to 10+ minutes.
- **Core functionality validation**:
  - `python -c "from game_rng import GameRNG; print('GameRNG available')"` -- validates core dependency
  - `python -c "import polars, numpy, numba; print('Core libs work')"` -- validates data processing stack
- **Run dungeon generation (WORKS RELIABLY)**:
  - `python main2.py --seed 12345 --max-nodes 50 --max-depth 10` -- small test (1-2 seconds)
  - `python main2.py --seed 12345 --max-nodes 200 --max-depth 30` -- medium test (3-5 seconds) 
  - NEVER CANCEL: Large dungeons can take 30+ seconds. Set timeout to 120+ seconds.
- **Run tests (PARTIAL SUCCESS)**:
  - `python -m pytest tests/ -v` -- runs all tests (3-5 seconds)
  - EXPECT: ~10 test failures due to API changes (missing entity_templates), 11 tests pass
  - Working tests validate: FOV, visibility, inventory, template spawning, line-of-sight

## Critical Limitations

- **GUI CANNOT RUN**: `python main.py` fails in headless environments (PySide6/Qt requires display)
- **Auto simulation BROKEN**: `./auto/run.sh` has Python syntax errors (missing imports)
- **Dungeon scripts FAIL**: `./Dungeon/run.sh` cannot find game_rng when run from subdirectory
- **Pathfinding tests FAIL**: Missing game module imports when run from subdirectory
- **Many tests FAIL**: GameState API changed, requires entity_templates parameter

## Validation Scenarios

ALWAYS run these validation steps after making changes:

1. **Core dependency check**: `python -c "from game_rng import GameRNG; rng = GameRNG(seed=123); print(f'RNG works: {rng.get_int(1,10)}')"` 
2. **Small dungeon generation**: `python main2.py --seed 42 --max-nodes 25 --max-depth 5` (should complete in <2 seconds)
3. **Run working tests**: `python -m pytest tests/test_fov_visibility.py tests/test_inventory_capacity.py tests/test_visibility.py -v` (should pass all tests)
4. **Import key modules**: `python -c "from game.game_state import GameState; from game.world.game_map import GameMap; print('Core imports work')"`

## Build Process

- **NO BUILD REQUIRED**: Python project, runs directly with interpreter
- **NO COMPILATION**: Uses Numba for JIT compilation at runtime
- **KEY FILES**: 
  - `main2.py` -- working dungeon generation pipeline  
  - `game_rng.py` -- custom RNG implementation (created to fix missing dependency)
  - `tests/` -- test suite with pytest
  - `config/` -- YAML configuration files

## Time Expectations

- **Dependency install**: 3-5 minutes (NEVER CANCEL - set 10+ minute timeout)
- **Small dungeon generation**: 1-2 seconds  
- **Medium dungeon generation**: 3-5 seconds
- **Large dungeon generation**: 30-60 seconds (NEVER CANCEL - set 120+ second timeout)
- **Test suite run**: 3-5 seconds
- **Individual test files**: <1 second each

## Environment Setup

- **Python version**: 3.11+ (tested with 3.12)
- **Core packages**: polars, numpy, numba, scipy, scikit-image
- **GUI packages**: pyside6, pillow (will not work in headless environments)
- **Dev packages**: pytest, joblib, structlog, pyyaml
- **NO CONDA REQUIRED**: Works with system Python and pip
- **Working directory**: Always run from repository root `/home/runner/work/basicrl/basicrl`

## Repository Structure

Key directories and their working status:

- `game/` -- Core game engine (IMPORTS WORK, some API changes)
- `tests/` -- Test suite (PARTIALLY WORKING - 11/21 tests pass)  
- `Dungeon/` -- Procedural generation (WORKS via main2.py, fails via run.sh)
- `auto/` -- AI simulation (BROKEN - syntax errors)
- `pathfinding/` -- Pathfinding algorithms (BROKEN - import errors)
- `config/` -- YAML configuration files (EXISTS)
- `utils/` -- Utilities and helpers (game_rng.py added)

## Common Issues and Workarounds

- **Import errors from subdirectories**: Run from repository root, not subdirectories
- **Missing game_rng**: Fixed by creating stub implementation in repository root
- **GUI failures in headless**: Use main2.py for dungeon generation instead of main.py
- **Test failures**: Expected due to API changes, focus on the 11 working tests
- **Module not found**: Ensure all dependencies installed with pip install command above
- **PYTHONPATH issues**: Always run from repository root directory

## Linting and Code Quality

- **NO LINTERS CONFIGURED**: Repository has no flake8, black, or mypy configuration
- **NO PRE-COMMIT HOOKS**: No automated formatting or linting
- **CODE STYLE**: Mixed style throughout codebase
- **TYPE HINTS**: Partial usage, inconsistent across modules

## Working Commands Reference

```bash
# Essential validation sequence
cd /home/runner/work/basicrl/basicrl
pip install polars numpy numba pyside6 pillow pyyaml structlog scipy scikit-image joblib pytest
python -c "from game_rng import GameRNG; print('Dependencies OK')"
python main2.py --seed 42 --max-nodes 25 --max-depth 5
python -m pytest tests/test_fov_visibility.py -v

# Medium complexity testing  
python main2.py --seed 12345 --max-nodes 100 --max-depth 20  # ~5-10 seconds
python -m pytest tests/ -k "visibility or inventory" -v     # Run working tests only

# Import validation
python -c "from game.game_state import GameState; print('GameState OK')"
python -c "from game.world.game_map import GameMap; print('GameMap OK')"
```

Always use these exact commands and expect the noted time ranges. If commands fail, check dependency installation first.