"""Stub AI system for tests.
Provides a no-op dispatch_ai function so game.state can import it."""

from __future__ import annotations


def dispatch_ai(*args, **kwargs):
    """Placeholder dispatcher used in tests."""
    return None
