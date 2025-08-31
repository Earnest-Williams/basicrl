"""Zone-based simulation scheduler.

This module tracks "zones" of the map and allows systems to queue updates
for zones that are far from the player.  Distant zones are updated at a lower
frequency which reduces the cost of simulating the entire world every turn.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Set, Tuple


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
        self.event_queue: Dict[Tuple[int, int], Callable[[Any], None]] = {}

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

    def schedule_zone_event(
        self, zone: Tuple[int, int], callback: Callable[[Any], None]
    ) -> None:
        """Queue ``callback`` to run for ``zone`` if none is already pending."""
        if zone not in self.event_queue:
            self.event_queue[zone] = callback

    def process(
        self, turn: int, active_zones: Set[Tuple[int, int]], game_state: Any
    ) -> None:
        """Process queued events.

        Events run immediately if their zone is active; otherwise they are
        executed after ``passive_interval`` turns have elapsed since the last
        update for that zone.
        """
        for zone, callback in list(self.event_queue.items()):
            if zone in active_zones or (
                turn - self.last_update[zone] >= self.passive_interval
            ):
                callback(game_state)
                self.last_update[zone] = turn
                del self.event_queue[zone]
