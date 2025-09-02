"""Package initialization for game effects.

On import this module registers any handlers defined in
``ART_SUBSTANCE_DISPATCHER`` with the ``magic.executor`` subsystem.  This
ensures that magical Works can resolve their effect functions before
execution.
"""

from .handlers import ART_SUBSTANCE_DISPATCHER
from magic.executor import register_handler


def _adapt_handler(handler):
    """Wrap legacy handlers to match ``(Work, GameState)`` signature."""

    def wrapper(work, game_state):
        context = {"game_state": game_state}
        handler(context, {})

    return wrapper


# Populate the magic executor's registry with all known art/substance mappings
for (art, substance), handler in ART_SUBSTANCE_DISPATCHER.items():
    register_handler(art, substance, _adapt_handler(handler))
