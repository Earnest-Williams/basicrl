# game/entities/registry.py
import polars as pl
from typing import Any, Self  # Keep Any for now, Self for return type

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
        self.entities_df: pl.DataFrame = pl.DataFrame(schema=ENTITY_SCHEMA)
        self._next_entity_id: int = 0  # Simple counter for unique IDs

    def _get_next_id(self: Self) -> int:
        """Generates the next available unique entity ID."""
        current_id = self._next_entity_id
        self._next_entity_id += 1
        # Check for potential overflow (unlikely with UInt32, but good practice)
        if self._next_entity_id > 2**32 - 1:
            # Consider alternatives: recycle IDs, use UInt64, or raise error
            raise OverflowError("Entity ID counter overflowed (UInt32 limit reached).")
        return current_id

    def create_entity(
        self: Self,
        x: int,
        y: int,
        glyph: int,  # Use integer Unicode codepoint
        color_fg: tuple[int, int, int],
        name: str,
        blocks_movement: bool = True,
        hp: int = 1,
        max_hp: int = 1,
        # **kwargs: Any # Allow adding extra components dynamically? Maybe later.
    ) -> int:
        """Creates a new entity, marks it as active, and adds it to the registry."""
        new_id = self._get_next_id()

        entity_data = {
            "entity_id": [new_id],
            "is_active": [True],  # New entities are active by default
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

        # Create a DataFrame for the new entity
        # Ensure schema includes all columns, even if defaults aren't specified here
        new_entity_df = pl.DataFrame(entity_data).cast(ENTITY_SCHEMA, strict=False)

        # Append the new entity DataFrame
        self.entities_df = self.entities_df.vstack(new_entity_df)

        return new_id

    def get_entity_component(
        self: Self, entity_id: int, component_name: str
    ) -> Any | None:
        """Retrieves the value of a specific component for a given *active* entity."""
        if component_name not in self.entities_df.columns:
            raise ValueError(f"Component '{component_name}' does not exist.")
        if component_name == "is_active":
            # Maybe allow getting this? For now, assume internal use.
            raise ValueError("Cannot directly get 'is_active' status via this method.")

        # Filter for the specific active entity and select the component
        result = (
            self.entities_df.lazy()
            .filter((pl.col("entity_id") == entity_id) & pl.col("is_active"))
            .select(component_name)
            .collect()
        )

        if result.height == 0:
            return None  # Entity not found or not active

        # Extract the single value using .item()
        # Polars ensures uniqueness if entity_id is a primary key,
        # but checking height first handles the 'not found / inactive' case.
        return result.item()

    def set_entity_component(
        self: Self, entity_id: int, component_name: str, value: Any
    ) -> bool:
        """Sets the value of a specific component for a given *active* entity."""
        if component_name not in self.entities_df.columns:
            raise ValueError(f"Component '{component_name}' does not exist.")
        if component_name in ("entity_id", "is_active"):
            raise ValueError(f"Cannot directly set '{component_name}' component.")

        # Check if the entity exists *and is active* before attempting update
        # Using lazy filter + head(1) + collect is efficient for existence check
        entity_active = (
            self.entities_df.lazy()
            .filter((pl.col("entity_id") == entity_id) & pl.col("is_active"))
            .head(1)
            .collect()
            .height
            > 0
        )

        if not entity_active:
            return False  # Return False if entity not found or inactive

        # Update the DataFrame using `with_columns` and `when/then/otherwise`
        # This creates a new DataFrame with the updated value. Only update active entity.
        self.entities_df = self.entities_df.with_columns(
            pl.when((pl.col("entity_id") == entity_id) & pl.col("is_active"))
            .then(pl.lit(value))  # Use pl.lit() to treat 'value' as a literal
            .otherwise(pl.col(component_name))  # Keep original value for other rows
            .alias(component_name)  # Ensure the column name remains the same
            .cast(
                ENTITY_SCHEMA[component_name], strict=False
            )  # Ensure type consistency
        )
        return True

    def get_position(self: Self, entity_id: int) -> tuple[int, int] | None:
        """Retrieves the (x, y) position of a given *active* entity."""
        result = (
            self.entities_df.lazy()
            .filter((pl.col("entity_id") == entity_id) & pl.col("is_active"))
            .select(["x", "y"])
            .collect()
        )

        if result.height == 0:
            return None
        return result.row(0)  # .row(0) efficiently gets the first row as a tuple

    def set_position(self: Self, entity_id: int, x: int, y: int) -> bool:
        """Sets the (x, y) position of a given *active* entity."""
        entity_active = (
            self.entities_df.lazy()
            .filter((pl.col("entity_id") == entity_id) & pl.col("is_active"))
            .head(1)
            .collect()
            .height
            > 0
        )

        if not entity_active:
            return False

        # Update 'x' and 'y' columns simultaneously for the active entity
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
        return True

    def get_entities_at(self: Self, x: int, y: int) -> pl.DataFrame:
        """Returns a DataFrame containing all *active* entities at the given coordinates."""
        return (
            self.entities_df.lazy()
            .filter((pl.col("x") == x) & (pl.col("y") == y) & pl.col("is_active"))
            .collect()
        )

    def get_blocking_entity_at(self: Self, x: int, y: int) -> int | None:
        """
        Finds the entity ID of the first *active*, blocking entity found at (x, y).
        Returns None if no such entity is present.
        """
        result = (
            self.entities_df.lazy()
            .filter(
                (pl.col("x") == x)
                & (pl.col("y") == y)
                & pl.col("blocks_movement")
                & pl.col("is_active")  # Assumes blocks_movement == True
            )
            .select("entity_id")
            .head(1)
            .collect()
        )

        if result.height > 0:
            return result.item()  # Return the entity_id
        return None

    def delete_entity(self: Self, entity_id: int) -> bool:
        """
        Marks an entity as inactive (soft delete). Returns True if successful.
        Does not physically remove the entity data until compact_registry() is called.
        """
        # Check if entity exists and is currently active before marking inactive
        entity_exists = (
            self.entities_df.lazy()
            .filter((pl.col("entity_id") == entity_id) & pl.col("is_active"))
            .head(1)
            .collect()
            .height
            > 0
        )

        if not entity_exists:
            return False  # Entity already inactive or doesn't exist

        # Set 'is_active' to False for the given entity_id
        self.entities_df = self.entities_df.with_columns(
            pl.when(pl.col("entity_id") == entity_id)
            .then(pl.lit(False))  # Mark as inactive
            .otherwise(pl.col("is_active"))
            .alias("is_active")
        )
        # Consider invalidating any caches here if they were implemented

        return True

    def compact_registry(self: Self) -> None:
        """
        Permanently removes all inactive entities from the DataFrame.
        This is a potentially costly operation and should be called periodically,
        not every frame or turn.
        """
        initial_count = self.entities_df.height
        self.entities_df = self.entities_df.filter(pl.col("is_active"))
        final_count = self.entities_df.height
        print(
            f"Registry compacted. Removed {initial_count - final_count} inactive entities."
        )
        # Rebuild any caches here if they were implemented

    def get_active_entities(self: Self) -> pl.DataFrame:
        """Returns a DataFrame containing only the active entities."""
        # Useful for iterating over all active entities in systems
        return self.entities_df.filter(pl.col("is_active"))
