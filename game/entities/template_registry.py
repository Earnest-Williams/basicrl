from __future__ import annotations

"""Registry for entity templates.

Stores immutable template data for entities and portals.  Templates are loaded
from configuration files at start-up and can be looked up by ID when spawning
or creating entities in the game world.
"""

from typing import Any, Dict, Self

import structlog

log = structlog.get_logger()


class EntityTemplateRegistry:
    """Simple container providing access to entity templates."""

    def __init__(self: Self, templates: Dict[str, Any] | None = None):
        self.templates: Dict[str, Any] = templates or {}
        log.debug(
            "EntityTemplateRegistry initialized", templates=len(self.templates)
        )

    def get_template(self: Self, template_id: str) -> Dict[str, Any] | None:
        """Retrieve a template definition by ID."""
        return self.templates.get(template_id)
