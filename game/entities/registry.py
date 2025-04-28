# game/entities/registry.py
import polars as pl
from typing import Any, Self
import structlog  # Added

log = structlog.get_logger()  # Added

# Define the schema for the entity DataFrame
ENTITY_SCHEMA: dict[str, pl.DataType] = {
    "entity_id": pl.UInt32,  # Unique identifier for the entity
    "is_active": pl.Boolean,  # Is the entity currently active? (for soft deletes)
    "x": pl.Int16,  # X-coordinate on the map
    "y": pl.Int16,  # Y-coordinate on the map
    "glyph": pl.UInt16,  # Unicode codepoint for rendering
    "color_fg_r": pl.UInt8,  # Foreground color (Red)
    "color_fg_g": pl.UInt8,  # Foreground color (Green)
    "color_fg_b": pl.UInt8,  # Foreground color (Blue)
    "name": pl.Utf8,  # Name of the entity (e.g., "Player", "Orc")
    "blocks_movement": pl.Boolean,  # Does this entity block movement?
    "hp": pl.Int16,  # Current hit points
    "max_hp": pl.Int16,  # Maximum hit points
    # Add more components as needed
}


class EntityRegistry:
    """
    Manages all entities and their components using a Polars DataFrame.
    Includes soft delete functionality via 'is_active' flag.
    """

    def __init__(self: Self):
        log.info("Initializing EntityRegistry")  # Added log
        self.entities_df: pl.DataFrame = pl.DataFrame(schema=ENTITY_SCHEMA)
        self._next_entity_id: int = 0
        log.debug(
            "EntityRegistry initialized with empty DataFrame", schema=ENTITY_SCHEMA
        )

    def _get_next_id(self: Self) -> int:
        """Generates the next available unique entity ID."""
        current_id = self._next_entity_id
        self._next_entity_id += 1
        # Check for potential overflow (unlikely with UInt32, but good practice)
        if self._next_entity_id > 2**32 - 1:
            # Log error before raising
            log.critical("Entity ID counter overflowed", next_id=self._next_entity_id)
            raise OverflowError("Entity ID counter overflowed (UInt32 limit reached).")
        return current_id

    def create_entity(
        self: Self,
        x: int,
        y: int,
        glyph: int,
        color_fg: tuple[int, int, int],
        name: str,
        blocks_movement: bool = True,
        hp: int = 1,
        max_hp: int = 1,
        # **kwargs: Any # Allow adding extra components dynamically? Maybe later.
    ) -> int:
        """Creates a new entity, marks it as active, and adds it to the registry."""
        new_id = self._get_next_id()
        log_context = {"name": name, "pos": (x, y), "glyph": glyph, "hp": hp}
        log.debug("Attempting to create entity", **log_context)

        entity_data = {
            "entity_id": [new_id],
            "is_active": [True],
            "x": [x],
            "y": [y],
            "glyph": [glyph],
            "color_fg_r": [color_fg[0]],
            "color_fg_g": [color_fg[1]],
            "color_fg_b": [color_fg[2]],
            "name": [name],
            "blocks_movement": [blocks_movement],
            "hp": [hp],
            "max_hp": [max_hp],
        }

        try:
            # Create a DataFrame for the new entity, casting ensures type safety
            new_entity_df = pl.DataFrame(entity_data).cast(ENTITY_SCHEMA, strict=False)
            # Append the new entity DataFrame
            self.entities_df = self.entities_df.vstack(new_entity_df)
            log.info("Entity created successfully", entity_id=new_id, **log_context)
            return new_id
        except Exception as e:
            log.error(
                "Failed to create entity DataFrame or vstack",
                error=str(e),
                exc_info=True,
                entity_data=entity_data,
            )
            # Consider how to handle this - potentially re-raise or return an error indicator?
            # For now, re-raising might be safest to halt inconsistent state.
            raise

    def get_entity_component(
        self: Self, entity_id: int, component_name: str
    ) -> Any | None:
        """Retrieves the value of a specific component for a given *active* entity."""
        log_context = {"entity_id": entity_id, "component": component_name}
        log.debug("Getting entity component", **log_context)

        if component_name not in self.entities_df.columns:
            log.warning("Component does not exist", **log_context)
            # Raising ValueError might be better than returning None if component is invalid
            raise ValueError(f"Component '{component_name}' does not exist.")
        if component_name == "is_active":
            log.warning(
                "Attempted to get 'is_active' component directly", **log_context
            )
            raise ValueError("Cannot directly get 'is_active' status via this method.")

        try:
            # Filter for the specific active entity and select the component
            result = (
                self.entities_df.lazy()
                .filter((pl.col("entity_id") == entity_id) & pl.col("is_active"))
                .select(component_name)
                .collect()
            )

            if result.height == 0:
                log.debug("Entity not found or inactive", **log_context)
                return None

            value = result.item()
            log.debug("Component retrieved successfully", value=value, **log_context)
            return value
        except Exception as e:
            log.error(
                "Error getting entity component",
                error=str(e),
                exc_info=True,
                **log_context,
            )
            return None  # Return None on unexpected error

    def set_entity_component(
        self: Self, entity_id: int, component_name: str, value: Any
    ) -> bool:
        """Sets the value of a specific component for a given *active* entity."""
        log_context = {
            "entity_id": entity_id,
            "component": component_name,
            "new_value": value,
        }
        log.debug("Setting entity component", **log_context)

        if component_name not in self.entities_df.columns:
            log.warning("Component does not exist", **log_context)
            raise ValueError(f"Component '{component_name}' does not exist.")
        if component_name in ("entity_id", "is_active"):
            log.warning("Attempted to set protected component", **log_context)
            raise ValueError(f"Cannot directly set '{component_name}' component.")

        try:
            # Check if the entity exists *and is active* before attempting update
            entity_active = (
                self.entities_df.lazy()
                .filter((pl.col("entity_id") == entity_id) & pl.col("is_active"))
                .select(pl.lit(1))  # Select a literal 1 to minimize data transfer
                .head(1)
                .collect()
                .height
                > 0
            )

            if not entity_active:
                log.debug(
                    "Entity not found or inactive, cannot set component", **log_context
                )
                return False

            # Update the DataFrame using `with_columns` and `when/then/otherwise`
            self.entities_df = self.entities_df.with_columns(
                pl.when((pl.col("entity_id") == entity_id) & pl.col("is_active"))
                .then(pl.lit(value))
                .otherwise(pl.col(component_name))
                .alias(component_name)
                .cast(
                    ENTITY_SCHEMA[component_name], strict=False
                )  # Ensure type consistency
            )
            log.debug("Component set successfully", **log_context)
            return True
        except Exception as e:
            log.error(
                "Error setting entity component",
                error=str(e),
                exc_info=True,
                **log_context,
            )
            return False

    def get_position(self: Self, entity_id: int) -> tuple[int, int] | None:
        """Retrieves the (x, y) position of a given *active* entity."""
        log_context = {"entity_id": entity_id, "component": "position"}
        # Use low level debug as position is frequently accessed
        # log.debug("Getting entity position", **log_context)
        try:
            result = (
                self.entities_df.lazy()
                .filter((pl.col("entity_id") == entity_id) & pl.col("is_active"))
                .select(["x", "y"])
                .collect()
            )

            if result.height == 0:
                # log.debug("Entity not found or inactive", **log_context)
                return None
            pos = result.row(0)
            # log.debug("Position retrieved successfully", position=pos, **log_context)
            return pos
        except Exception as e:
            log.error(
                "Error getting entity position",
                error=str(e),
                exc_info=True,
                **log_context,
            )
            return None

    def set_position(self: Self, entity_id: int, x: int, y: int) -> bool:
        """Sets the (x, y) position of a given *active* entity."""
        log_context = {
            "entity_id": entity_id,
            "component": "position",
            "new_value": (x, y),
        }
        # Use low level debug as position is frequently set
        # log.debug("Setting entity position", **log_context)
        try:
            # Check if active first
            entity_active = (
                self.entities_df.lazy()
                .filter((pl.col("entity_id") == entity_id) & pl.col("is_active"))
                .select(pl.lit(1))
                .head(1)
                .collect()
                .height
                > 0
            )

            if not entity_active:
                # log.debug("Entity not found or inactive, cannot set position", **log_context)
                return False

            # Update 'x' and 'y' columns simultaneously
            self.entities_df = self.entities_df.with_columns(
                [
                    pl.when((pl.col("entity_id") == entity_id) & pl.col("is_active"))
                    .then(pl.lit(x).cast(pl.Int16))
                    .otherwise(pl.col("x"))
                    .alias("x"),
                    pl.when((pl.col("entity_id") == entity_id) & pl.col("is_active"))
                    .then(pl.lit(y).cast(pl.Int16))
                    .otherwise(pl.col("y"))
                    .alias("y"),
                ]
            )
            # log.debug("Position set successfully", **log_context)
            return True
        except Exception as e:
            log.error(
                "Error setting entity position",
                error=str(e),
                exc_info=True,
                **log_context,
            )
            return False

    def get_entities_at(self: Self, x: int, y: int) -> pl.DataFrame:
        """Returns a DataFrame containing all *active* entities at the given coordinates."""
        # log.debug("Getting entities at position", pos=(x,y)) # Can be noisy
        try:
            return (
                self.entities_df.lazy()
                .filter((pl.col("x") == x) & (pl.col("y") == y) & pl.col("is_active"))
                .collect()
            )
        except Exception as e:
            log.error(
                "Error getting entities at position",
                error=str(e),
                exc_info=True,
                pos=(x, y),
            )
            # Return empty dataframe on error to avoid breaking callers expecting a DataFrame
            return pl.DataFrame(schema=ENTITY_SCHEMA)

    def get_blocking_entity_at(self: Self, x: int, y: int) -> int | None:
        """
        Finds the entity ID of the first *active*, blocking entity found at (x, y).
        Returns None if no such entity is present.
        """
        log_context = {"pos": (x, y)}
        # log.debug("Getting blocking entity at position", **log_context) # Can be noisy
        try:
            result = (
                self.entities_df.lazy()
                .filter(
                    (pl.col("x") == x)
                    & (pl.col("y") == y)
                    & pl.col("blocks_movement")  # Assumes blocks_movement == True check
                    & pl.col("is_active")
                )
                .select("entity_id")
                .head(1)  # Optimization: only need one if it exists
                .collect()
            )

            if result.height > 0:
                entity_id = result.item()
                # log.debug("Blocking entity found", entity_id=entity_id, **log_context)
                return entity_id
            # log.debug("No blocking entity found", **log_context)
            return None
        except Exception as e:
            log.error(
                "Error getting blocking entity",
                error=str(e),
                exc_info=True,
                **log_context,
            )
            return None

    def delete_entity(self: Self, entity_id: int) -> bool:
        """
        Marks an entity as inactive (soft delete). Returns True if successful.
        Does not physically remove the entity data until compact_registry() is called.
        """
        log_context = {"entity_id": entity_id}
        log.debug("Deleting entity (marking inactive)", **log_context)
        try:
            # Check if entity exists and is currently active before marking inactive
            entity_exists = (
                self.entities_df.lazy()
                .filter((pl.col("entity_id") == entity_id) & pl.col("is_active"))
                .select(pl.lit(1))
                .head(1)
                .collect()
                .height
                > 0
            )

            if not entity_exists:
                log.debug("Entity already inactive or does not exist", **log_context)
                return False

            # Set 'is_active' to False for the given entity_id
            self.entities_df = self.entities_df.with_columns(
                pl.when(pl.col("entity_id") == entity_id)
                .then(pl.lit(False))
                .otherwise(pl.col("is_active"))
                .alias("is_active")
            )
            log.info("Entity marked as inactive", **log_context)
            return True
        except Exception as e:
            log.error(
                "Error deleting entity (marking inactive)",
                error=str(e),
                exc_info=True,
                **log_context,
            )
            return False

    def compact_registry(self: Self) -> None:
        """
        Permanently removes all inactive entities from the DataFrame.
        This is a potentially costly operation and should be called periodically,
        not every frame or turn.
        """
        log.info("Compacting entity registry...")
        try:
            initial_count = self.entities_df.height
            self.entities_df = self.entities_df.filter(pl.col("is_active"))
            final_count = self.entities_df.height
            removed_count = initial_count - final_count
            # Replaced print with log.info
            log.info(
                "Registry compacted",
                initial_count=initial_count,
                final_count=final_count,
                removed_count=removed_count,
            )
        except Exception as e:
            log.error("Error compacting registry", error=str(e), exc_info=True)

    def get_active_entities(self: Self) -> pl.DataFrame:
        """Returns a DataFrame containing only the active entities."""
        # Useful for iterating over all active entities in systems
        # log.debug("Getting active entities DataFrame") # Very noisy
        try:
            return self.entities_df.filter(pl.col("is_active"))
        except Exception as e:
            log.error("Error getting active entities", error=str(e), exc_info=True)
            return pl.DataFrame(schema=ENTITY_SCHEMA)  # Return empty on error
