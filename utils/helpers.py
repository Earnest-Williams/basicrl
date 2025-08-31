# utils/helpers.py
import re
from typing import TYPE_CHECKING, Union # *** MODIFIED: Added Union ***

import structlog  # Added for logging potential errors

# Assuming GameRNG might be defined elsewhere, handle potential import
try:
    # Adjust import path based on actual location of GameRNG
    from game_rng import GameRNG
except ImportError:
    # Fallback if GameRNG is not directly importable this way
    # This might happen if game_rng.py is not in python path correctly
    # Or if it's intended to be passed explicitly always.
    # For now, we add a check in _roll_dice.
    GameRNG = None

if TYPE_CHECKING:
    # Conditional import for type checkers
    if GameRNG is None:
        from typing import Any

        GameRNG = Any  # Provide a fallback type hint

log = structlog.get_logger(__name__)  # Use module logger

# --- Dice Rolling Utility (Moved from effects.handlers) ---
DICE_PATTERN = re.compile(r"(\d+)?d(\d+)(?:([+-])(\d+))?")


# Make it a public function
# *** MODIFIED: Replaced | with Union[] ***
def roll_dice(dice_str: Union[str, None], rng: Union["GameRNG", None]) -> int:
    """
    Rolls dice based on a string format (e.g., '1d6', '2d4+1').
    Requires a GameRNG instance.
    """
    if not dice_str:
        return 0
    if rng is None:
        # Check if GameRNG was imported successfully, otherwise raise error
        if GameRNG is None:
            log.critical(
                "GameRNG type could not be imported, cannot roll dice without RNG instance."
            )
            raise TypeError("GameRNG instance is required for roll_dice.")
        else:
            log.error("Dice roll attempted without RNG instance!")
            # Consider raising an error instead of returning 0? Depends on expected usage.
            # raise ValueError("RNG instance is required for roll_dice.")
            return 0  # Returning 0 for now, but this indicates an issue.

    match = DICE_PATTERN.match(dice_str)
    if match:
        num_dice_str, sides_str, operator, bonus_str = match.groups()
        num_dice = int(num_dice_str) if num_dice_str else 1
        sides = int(sides_str)
        bonus = int(f"{operator}{bonus_str}") if operator and bonus_str else 0
        if sides <= 0:
            return bonus
        if num_dice <= 0:
            return bonus
        # Use the passed RNG instance
        try:
            roll_total = sum(rng.get_int(1, sides) for _ in range(num_dice))
            return roll_total + bonus
        except AttributeError:
            log.error(
                "Passed rng object does not have expected 'get_int' method.",
                rng_type=type(rng),
            )
            raise  # Re-raise the error as this is unexpected
        except Exception as e:
            log.error("Error during RNG dice roll", error=str(e), exc_info=True)
            raise  # Re-raise other RNG errors

    else:
        try:
            return int(dice_str)  # Allow plain numbers
        except ValueError:
            log.error("Invalid dice string format", dice_str=dice_str)
            return 0
