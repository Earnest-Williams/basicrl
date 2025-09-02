"""Field-of-view calculations using symmetrical shadowcasting.

This module defines the :class:`MyVisibility` class which performs a field of
view computation given user supplied callables for blocking checks,
visibility writes and distance calculations.
"""

from __future__ import annotations

from typing import Callable


class MyVisibility:
    """Generic visibility calculator using symmetrical shadowcasting.

    Parameters
    ----------
    blocks_light:
        Callable receiving ``(x, y)`` returning ``True`` if the tile blocks
        light or lies outside the map bounds.
    set_visible:
        Callable receiving ``(x, y)`` which marks the tile as visible.  This is
        also a suitable place to flag tiles as explored.
    get_distance:
        Callable receiving the relative ``(dx, dy)`` from the origin and
        returning the distance value used to clamp the search radius.
    """

    def __init__(
        self,
        *,
        blocks_light: Callable[[int, int], bool],
        set_visible: Callable[[int, int], None],
        get_distance: Callable[[int, int], float],
    ) -> None:
        self.blocks_light = blocks_light
        self.set_visible = set_visible
        self.get_distance = get_distance

    def compute(self, origin_x: int, origin_y: int, radius: int) -> None:
        """Compute visibility from ``(origin_x, origin_y)`` within ``radius``."""

        self.set_visible(origin_x, origin_y)
        for octant in range(8):
            self._cast_light(
                origin_x, origin_y, 1, 1.0, 0.0, radius, *self._multipliers[octant]
            )

    # Transformation coefficients for the eight octants.
    _multipliers = (
        (1, 0, 0, 1),
        (0, 1, 1, 0),
        (0, -1, 1, 0),
        (-1, 0, 0, 1),
        (-1, 0, 0, -1),
        (0, -1, -1, 0),
        (0, 1, -1, 0),
        (1, 0, 0, -1),
    )

    def _cast_light(
        self,
        cx: int,
        cy: int,
        row: int,
        start_slope: float,
        end_slope: float,
        radius: int,
        xx: int,
        xy: int,
        yx: int,
        yy: int,
    ) -> None:
        """Recursively cast light in a single octant."""

        if start_slope < end_slope:
            return

        radius * radius
        for j in range(row, radius + 1):
            dx = -j
            dy = -j
            blocked = False
            new_start = start_slope
            while dx <= 0:
                dx += 1
                mx = cx + dx * xx + dy * xy
                my = cy + dx * yx + dy * yy

                l_slope = (dx - 0.5) / (dy + 0.5)
                r_slope = (dx + 0.5) / (dy - 0.5)
                if start_slope < r_slope:
                    continue
                if end_slope > l_slope:
                    break

                if self.get_distance(mx - cx, my - cy) <= radius:
                    self.set_visible(mx, my)

                if blocked:
                    if self.blocks_light(mx, my):
                        new_start = r_slope
                        continue
                    else:
                        blocked = False
                        start_slope = new_start
                else:
                    if self.blocks_light(mx, my) and j < radius:
                        blocked = True
                        self._cast_light(
                            cx,
                            cy,
                            j + 1,
                            start_slope,
                            l_slope,
                            radius,
                            xx,
                            xy,
                            yx,
                            yy,
                        )
                        new_start = r_slope
            if blocked:
                break
