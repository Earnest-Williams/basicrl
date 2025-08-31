"""Zone-based simulation scheduler.

This module tracks "zones" of the map and allows systems to queue updates
for zones that are far from the player.  Distant zones are updated at a lower
frequency which reduces the cost of simulating the entire world every turn.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Set, Tuple
import importlib


def _resolve_callback(path: str) -> Callable[[Any], None] | None:
    """Resolve ``path`` into a callable.

    ``path`` should be of the form ``"module.submodule:qualname"``.  If the
    import fails or the attribute does not exist, ``None`` is returned.
    """
    try:
        module_name, qualname = path.split(":", 1)
        module = importlib.import_module(module_name)
        obj: Any = module
        for attr in qualname.split("."):
            obj = getattr(obj, attr)
        if callable(obj):
            return obj  # type: ignore[return-value]
    except Exception:
        return None
    return None


class ZoneManager:
    """Manage coarse simulation zones.

    The map is divided into square zones.  Each zone can have a single
    callback queued which will be invoked when the zone becomes active or
    after a configurable number of turns if it remains distant.
    """

    def __init__(
        self,
        map_width: int,
        map_height: int,
        zone_size: int = 16,
        active_radius: int = 2,
        passive_interval: int = 5,
    ) -> None:
        self.map_width = map_width
        self.map_height = map_height
        self.zone_size = zone_size
        self.active_radius = active_radius
        self.passive_interval = passive_interval
        self.last_update: Dict[Tuple[int, int], int] = defaultdict(int)
        # Queue stores mapping of zone -> event id.  Event callbacks are kept in
        # ``event_registry`` so they can be serialised via identifier and
        # restored on load.
        self.event_queue: Dict[Tuple[int, int], int] = {}
        self.event_registry: Dict[int, Callable[[Any], None]] = {}
        self._next_event_id: int = 1

    # ------------------------------------------------------------------
    # Zone helpers
    # ------------------------------------------------------------------
    def get_zone(self, x: int, y: int) -> Tuple[int, int]:
        """Return the zone coordinate for the tile ``(x, y)``."""
        return x // self.zone_size, y // self.zone_size

    def get_active_zones(self, player_pos: Tuple[int, int] | None) -> Set[Tuple[int, int]]:
        """Return the set of zones considered active around ``player_pos``."""
        if player_pos is None:
            return set()
        px, py = player_pos
        pz_x, pz_y = self.get_zone(px, py)
        zones: Set[Tuple[int, int]] = set()
        for dx in range(-self.active_radius, self.active_radius + 1):
            for dy in range(-self.active_radius, self.active_radius + 1):
                zones.add((pz_x + dx, pz_y + dy))
        return zones

    # ------------------------------------------------------------------
    # Event scheduling and processing
    # ------------------------------------------------------------------
    def schedule_event(
        self, x: int, y: int, callback: Callable[[Any], None]
    ) -> None:
        """Schedule a callback for the zone containing ``(x, y)``."""
        zone = self.get_zone(x, y)
        self.schedule_zone_event(zone, callback)

    def _register_event(self, callback: Callable[[Any], None]) -> int:
        """Register ``callback`` and return a new event identifier."""
        event_id = self._next_event_id
        self._next_event_id += 1
        self.event_registry[event_id] = callback
        return event_id

    def schedule_zone_event(
        self, zone: Tuple[int, int], callback: Callable[[Any], None]
    ) -> None:
        """Queue ``callback`` to run for ``zone`` if none is already pending."""
        if zone not in self.event_queue:
            self.event_queue[zone] = self._register_event(callback)

    def process(
        self, turn: int, active_zones: Set[Tuple[int, int]], game_state: Any
    ) -> None:
        """Process queued events.

        Events run immediately if their zone is active; otherwise they are
        executed after ``passive_interval`` turns have elapsed since the last
        update for that zone.
        """
        for zone, event_id in list(self.event_queue.items()):
            if zone in active_zones or (
                turn - self.last_update[zone] >= self.passive_interval
            ):
                callback = self.event_registry.get(event_id)
                if callback:
                    callback(game_state)
                self.last_update[zone] = turn
                del self.event_queue[zone]
                self.event_registry.pop(event_id, None)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialise the manager to a plain dictionary."""
        return {
            "map_width": self.map_width,
            "map_height": self.map_height,
            "zone_size": self.zone_size,
            "active_radius": self.active_radius,
            "passive_interval": self.passive_interval,
            "last_update": [
                [zx, zy, turn] for (zx, zy), turn in self.last_update.items()
            ],
            "event_queue": [
                [zx, zy, event_id] for (zx, zy), event_id in self.event_queue.items()
            ],
            "event_registry": [
                [event_id, f"{cb.__module__}:{cb.__qualname__}"]
                for event_id, cb in self.event_registry.items()
            ],
            "next_event_id": self._next_event_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ZoneManager":
        """Restore a ``ZoneManager`` from ``data``.

        Event callbacks are resolved from their fully qualified dotted paths
        stored in ``data['event_registry']``.  If a callback cannot be
        imported it is skipped, leaving the corresponding event id without a
        callable, which will be ignored during processing.
        """
        manager = cls(
            data.get("map_width", 0),
            data.get("map_height", 0),
            zone_size=data.get("zone_size", 16),
            active_radius=data.get("active_radius", 2),
            passive_interval=data.get("passive_interval", 5),
        )
        manager.last_update = defaultdict(
            int,
            {
                (zx, zy): turn
                for zx, zy, turn in data.get("last_update", [])
            },
        )
        manager.event_queue = {
            (zx, zy): event_id
            for zx, zy, event_id in data.get("event_queue", [])
        }
        manager.event_registry = {}
        for event_id, path in data.get("event_registry", []):
            callback = _resolve_callback(path)
            if callback:
                manager.event_registry[event_id] = callback
        manager._next_event_id = data.get("next_event_id", 1)
        return manager
