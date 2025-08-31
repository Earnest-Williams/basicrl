# game/entities/registry.py
from typing import Any, Dict, List, Self

import polars as pl
import structlog

from game.entities.components import CombatStats, Inventory, Position, Renderable

try:
    from game.items.registry import BodySlotType, EquipSlot
except ImportError:
    log = structlog.get_logger()
    log.error(
        "CRITICAL: Could not import EquipSlot/BodySlotType from game.items.registry."
    )
    EquipSlot = Any
    BodySlotType = Any

log = structlog.get_logger()

# ENTITY_SCHEMA (Unchanged)
ENTITY_SCHEMA: dict[str, pl.DataType] = {
    "entity_id": pl.UInt32,
    "is_active": pl.Boolean,
    "x": pl.Int16,
    "y": pl.Int16,
    "glyph": pl.UInt16,
    "color_fg_r": pl.UInt8,
    "color_fg_g": pl.UInt8,
    "color_fg_b": pl.UInt8,
    "name": pl.Utf8,
    "ai_type": pl.Utf8,
    "blocks_movement": pl.Boolean,
    "hp": pl.Int16,
    "max_hp": pl.Int16,
    "inventory_capacity": pl.UInt16,
    "status_effects": pl.List(
        pl.Struct({"id": pl.Utf8, "duration": pl.Int16, "intensity": pl.Float32})
    ),
    "mana": pl.Float32,
    "max_mana": pl.Float32,
    "fullness": pl.Float32,
    "max_fullness": pl.Float32,
    "equipped_item_ids": pl.List(pl.UInt64),
    "body_plan": pl.Object,
}


class EntityRegistry:
    def __init__(self: Self):
        # (Initialization unchanged)
        log.info("Initializing EntityRegistry with body_plan schema")
        global ENTITY_SCHEMA
        if "equipped_item_ids" in ENTITY_SCHEMA and not isinstance(
            ENTITY_SCHEMA["equipped_item_ids"], pl.List
        ):
            log.warning("Correcting equipped_item_ids dtype in ENTITY_SCHEMA")
            ENTITY_SCHEMA["equipped_item_ids"] = pl.List(pl.UInt64)
        if "status_effects" in ENTITY_SCHEMA and not isinstance(
            ENTITY_SCHEMA["status_effects"], pl.List
        ):
            log.warning("Correcting status_effects dtype in ENTITY_SCHEMA")
            ENTITY_SCHEMA["status_effects"] = pl.List(
                pl.Struct(
                    {"id": pl.Utf8, "duration": pl.Int16, "intensity": pl.Float32}
                )
            )
        self.entities_df: pl.DataFrame = pl.DataFrame(schema=ENTITY_SCHEMA)
        self._next_entity_id: int = 0
        log.debug("EntityRegistry initialized", schema=list(ENTITY_SCHEMA.keys()))

    def _get_next_id(self: Self) -> int:
        # (Implementation unchanged)
        current_id = self._next_entity_id
        self._next_entity_id += 1
        if self._next_entity_id > 2**32 - 1:
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
        ai_type: str | None = None,
        hp: int = 1,
        max_hp: int = 1,
        inventory_capacity: int = 26,
        mana: float = 0.0,
        max_mana: float = 0.0,
        fullness: float = 100.0,
        max_fullness: float = 100.0,
        status_effects: list | None = None,
        initial_body_plan: Dict[str, int] | None = None,
    ) -> int:
        # (Implementation unchanged - uses direct schema on creation)
        new_id = self._get_next_id()
        log_context = {"name": name, "pos": (x, y), "glyph": glyph, "hp": hp}
        log.debug("Attempting to create entity", **log_context)
        default_body_plan = {
            "finger": 10,
            "hand": 2,
            "wrist": 2,
            "head": 1,
            "face": 1,
            "eyes": 1,
            "neck": 2,
            "upper_body_inner": 1,
            "upper_body_outer": 1,
            "lower_body_inner": 1,
            "lower_body_outer": 1,
            "feet": 2,
            "back": 1,
            "belt": 1,
        }
        body_plan = default_body_plan.copy()
        if initial_body_plan:
            if BodySlotType is Any:
                validated_initial = initial_body_plan
            else:
                try:
                    valid_keys = BodySlotType.__args__
                    validated_initial = {
                        k: v for k, v in initial_body_plan.items() if k in valid_keys
                    }
                    if len(validated_initial) != len(initial_body_plan):
                        log.warning(
                            "Invalid keys found in initial_body_plan, ignored.",
                            invalid_keys=set(initial_body_plan.keys())
                            - set(valid_keys),
                        )
                except AttributeError:
                    log.warning(
                        "BodySlotType definition not available, cannot validate initial_body_plan keys."
                    )
                    validated_initial = initial_body_plan
            body_plan.update(validated_initial)

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
            "ai_type": [ai_type],
            "blocks_movement": [blocks_movement],
            "hp": [hp],
            "max_hp": [max_hp],
            "inventory_capacity": [inventory_capacity],
            "status_effects": [status_effects if status_effects is not None else []],
            "mana": [mana],
            "max_mana": [max_mana],
            "fullness": [fullness],
            "max_fullness": [max_fullness],
            "equipped_item_ids": [[]],
            "body_plan": [body_plan],
        }
        try:
            new_entity_df = pl.DataFrame(entity_data, schema=ENTITY_SCHEMA)
            if self.entities_df.height == 0:
                self.entities_df = new_entity_df
            else:
                self.entities_df = pl.concat(
                    [self.entities_df, new_entity_df.select(self.entities_df.columns)],
                    how="vertical",
                )
            log.info("Entity created successfully", entity_id=new_id, **log_context)
            return new_id
        except Exception as e:
            log.error(
                "Failed to create entity DataFrame or append",
                error=str(e),
                exc_info=True,
                entity_data=entity_data,
            )
            raise

    def get_entity_component(
        self: Self, entity_id: int, component_name: str
    ) -> Any | None:
        """Retrieves the value of a specific component for a given *active* entity."""
        log_context = {"entity_id": entity_id, "component": component_name}
        if component_name not in self.entities_df.columns:
            log.warning("Component does not exist", **log_context)
            raise ValueError(
                f"Component '{component_name}' does not exist in ENTITY_SCHEMA."
            )

        try:
            # Filter first
            entity_df = self.entities_df.filter(
                (pl.col("entity_id") == entity_id) & pl.col("is_active")
            )

            if entity_df.height == 0:
                return None

            # Select the single column *after* filtering
            result_series = entity_df.select(component_name)[
                component_name
            ]  # Select column Series

            if (
                result_series.is_empty()
            ):  # Should not happen if height > 0, but safety check
                return None

            # --- MODIFICATION START ---
            # Check the target data type from the schema
            target_dtype = ENTITY_SCHEMA[component_name]

            # Extract value, handling List types specifically
            if isinstance(target_dtype, pl.List):
                # For list types, accessing the first element of the Series
                # and then converting to list should yield the Python list.
                # Polars Series.item() often returns the first element directly.
                list_value = result_series.item()  # Get the list element
                # Ensure it's actually a list (it should be if schema is correct)
                return (
                    list_value if isinstance(list_value, list) else []
                )  # Return empty list if type mismatch
            else:
                # For other types, .item() should return the scalar value
                return result_series.item()
            # --- MODIFICATION END ---

        except Exception as e:
            log.error(
                "Error getting entity component",
                error=str(e),
                exc_info=True,
                **log_context,
            )
            return None

    def set_entity_component(
        self: Self, entity_id: int, component_name: str, value: Any
    ) -> bool:
        # (Implementation unchanged)
        log_context = {
            "entity_id": entity_id,
            "component": component_name,
            "new_value": value,
        }
        if component_name not in self.entities_df.columns:
            log.warning("Component does not exist", **log_context)
            raise ValueError(
                f"Component '{component_name}' does not exist in ENTITY_SCHEMA."
            )
        if component_name in ("entity_id", "is_active"):
            log.warning("Attempted to set protected component", **log_context)
            raise ValueError(f"Cannot directly set '{component_name}' component.")
        try:
            target_dtype = ENTITY_SCHEMA[component_name]
            try:
                if target_dtype == pl.Object and not isinstance(
                    value, (dict, list, type(None))
                ):
                    log.warning(
                        f"Potentially incompatible type for Object column '{component_name}'",
                        type=type(value),
                        **log_context,
                    )
                lit_value = pl.lit(value).cast(target_dtype, strict=False)
            except Exception as cast_err:
                log.error(
                    f"Type error setting component '{component_name}'. Expected compatible with {target_dtype}.",
                    value_type=type(value),
                    error=cast_err,
                    **log_context,
                )
                return False
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
                log.debug(
                    "Entity not found or inactive, cannot set component", **log_context
                )
                return False
            self.entities_df = self.entities_df.with_columns(
                pl.when((pl.col("entity_id") == entity_id) & pl.col("is_active"))
                .then(lit_value)
                .otherwise(pl.col(component_name))
                .alias(component_name)
                .cast(target_dtype, strict=False)
            )
            return True
        except Exception as e:
            log.error(
                "Error setting entity component",
                error=str(e),
                exc_info=True,
                **log_context,
            )
            return False

    def get_position(self: Self, entity_id: int) -> Position | None:
        """Return the Position component for an entity if available."""
        pos_x = self.get_entity_component(entity_id, "x")
        pos_y = self.get_entity_component(entity_id, "y")
        if pos_x is not None and pos_y is not None:
            return Position(int(pos_x), int(pos_y))
        return None

    def set_position(self: Self, entity_id: int, position: Position) -> bool:
        """Update an entity's position component."""
        success_x = self.set_entity_component(entity_id, "x", position.x)
        success_y = self.set_entity_component(entity_id, "y", position.y)
        return success_x and success_y

    def get_entities_at(self: Self, x: int, y: int) -> pl.DataFrame:
        # (Implementation unchanged)
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
            return pl.DataFrame(schema=ENTITY_SCHEMA)

    def get_blocking_entity_at(self: Self, x: int, y: int) -> int | None:
        # (Implementation unchanged)
        try:
            result = (
                self.entities_df.lazy()
                .filter(
                    (pl.col("x") == x)
                    & (pl.col("y") == y)
                    & pl.col("blocks_movement")
                    & pl.col("is_active")
                )
                .select("entity_id")
                .head(1)
                .collect()
            )
            if result.height > 0:
                return result.item()
            return None
        except Exception as e:
            log.error(
                "Error getting blocking entity", error=str(e), exc_info=True, pos=(x, y)
            )
            return None

    def delete_entity(self: Self, entity_id: int) -> bool:
        # (Implementation unchanged)
        log_context = {"entity_id": entity_id}
        log.debug("Deleting entity (marking inactive)", **log_context)
        try:
            entity_active_mask = (pl.col("entity_id") == entity_id) & pl.col(
                "is_active"
            )
            if self.entities_df.filter(entity_active_mask).height == 0:
                log.debug("Entity already inactive or does not exist", **log_context)
                return False
            self.entities_df = self.entities_df.with_columns(
                pl.when(entity_active_mask)
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
        # (Implementation unchanged)
        log.info("Compacting entity registry...")
        try:
            initial_count = self.entities_df.height
            self.entities_df = self.entities_df.filter(pl.col("is_active"))
            final_count = self.entities_df.height
            removed_count = initial_count - final_count
            log.info(
                "Registry compacted",
                initial_count=initial_count,
                final_count=final_count,
                removed_count=removed_count,
            )
        except Exception as e:
            log.error("Error compacting registry", error=str(e), exc_info=True)

    def get_active_entities(self: Self) -> pl.DataFrame:
        # (Implementation unchanged)
        try:
            return self.entities_df.filter(pl.col("is_active"))
        except Exception as e:
            log.error("Error getting active entities", error=str(e), exc_info=True)
            return pl.DataFrame(schema=ENTITY_SCHEMA)

    def get_body_plan(self, entity_id: int) -> Dict[str, int] | None:
        # (Implementation unchanged)
        plan = self.get_entity_component(entity_id, "body_plan")
        return plan if isinstance(plan, dict) else None

    def set_body_plan(self, entity_id: int, body_plan: Dict[str, int]) -> bool:
        # (Implementation unchanged)
        if not isinstance(body_plan, dict):
            log.error(
                "Invalid body_plan type provided to set_body_plan",
                entity_id=entity_id,
                type=type(body_plan),
            )
            return False
        return self.set_entity_component(entity_id, "body_plan", body_plan)

    def get_equipped_ids(self, entity_id: int) -> List[int] | None:
        """Gets the list of directly equipped item IDs for an active entity."""
        ids = self.get_entity_component(entity_id, "equipped_item_ids")
        # --- REMOVED Series check ---
        # The get_entity_component method should now handle returning a list directly.
        # Ensure a list is returned (or None if component retrieval failed)
        return ids if isinstance(ids, list) else None  # Return None if not a list

    def set_equipped_ids(self, entity_id: int, equipped_ids: List[int]) -> bool:
        # (Implementation unchanged)
        if not isinstance(equipped_ids, list):
            log.error(
                "Invalid equipped_ids type provided to set_equipped_ids",
                entity_id=entity_id,
                type=type(equipped_ids),
            )
            return False
        return self.set_entity_component(entity_id, "equipped_item_ids", equipped_ids)
