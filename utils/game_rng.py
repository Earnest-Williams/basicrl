from game_rng import GameRNG as _GameRNG


class GameRNG(_GameRNG):
    """Thin wrapper exposing missing helpers for legacy imports."""

    def choice(self, seq):  # type: ignore[override]
        """Return a random element from *seq*."""
        return seq[self.get_int(0, len(seq) - 1)]


__all__ = ["GameRNG"]
