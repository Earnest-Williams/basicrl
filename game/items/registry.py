# game/items/registry.py
from typing import Any, Dict, Literal, Self, cast

import polars as pl
import structlog

log = structlog.get_logger()

# --- Expanded EquipSlot Definition ---
# Specific slots for ITEM definitions (where an item CAN go)
EquipSlot = Literal[
    "upper_body_inner",
    "upper_body_outer",
    "lower_body_inner",
    "lower_body_outer",
    "head",
    "face",
    "eyes",
    "feet",
    "wrists",
    "main_hand",
    "off_hand",
    "finger_0",
    "finger_1",
    "finger_2",
    "finger_3",
    "finger_4",
    "finger_5",
    "finger_6",
    "finger_7",
    "finger_8",
    "finger_9",
    "neck_1",
    "neck_2",
    "belt",
    "back",
    # "held_light", # REMOVED
    "generic_attachment_point",  # Example
]
# --- End Expanded Definition ---

# General types used in ENTITY body_plan
BodySlotType = Literal[
    "upper_body_inner",
    "upper_body_outer",
    "lower_body_inner",
    "lower_body_outer",
    "head",
    "face",
    "eyes",
    "feet",
    "wrist",
    "hand",
    "finger",
    "neck",
    "belt",
    "back",
]

ItemLocation = Literal[
    "ground", "inventory", "equipped", "attached"
]  # Added "attached"

# Define the schema for the item DataFrame
ITEM_SCHEMA: dict[str, pl.DataType] = {
    "item_id": pl.UInt64,
    "is_active": pl.Boolean,
    "template_id": pl.Enum([]),  # Categories added in __init__
    "name": pl.Utf8,
    "glyph": pl.UInt16,
    "color_fg_r": pl.UInt8,
    "color_fg_g": pl.UInt8,
    "color_fg_b": pl.UInt8,
    # --- Location and Ownership ---
    "location_type": pl.Enum(ItemLocation.__args__),
    # Nullable. ID of entity holding in inv/equipped DIRECTLY
    "owner_entity_id": pl.UInt32,
    "x": pl.Int16,  # Nullable
    "y": pl.Int16,  # Nullable
    # --- Equip/Attach State ---
    "equipped_slot": pl.Enum(
        list(EquipSlot.__args__)
        # Nullable. Stores the SPECIFIC slot used (e.g. finger_3, neck_1) when location="equipped"
    ),
    # -- Mount Points (on Host Item) --
    # List[Dict{'id': str, 'compatible_types': List[str], 'accepted_item_id': int | None, 'flags': List[str]}]
    "mount_points": pl.Object,  # Nullable
    # -- Attachable Info (on Attachable Item) --
    # Dict{'compatible_mount_types': List[str], 'flags_required': List[str]}
    "attachable_info": pl.Object,  # Nullable
    # -- Attachment Link (on Attached Item) --
    # Nullable. ID of the item it's attached TO
    "parent_attachment_item_id": pl.UInt64,
    # Nullable. 'id' of the mount point used on parent
    "parent_attachment_slot_id": pl.Utf8,
    # -- Other Components --
    "quantity": pl.UInt16,
    "current_charge": pl.Int16,  # Nullable
    "current_fuel": pl.Int16,  # Nullable. Example for lanterns
    "current_poison_doses": pl.UInt8,  # Nullable. Example
    "current_durability": pl.Float32,  # Nullable
    "is_identified": pl.Boolean,
    "is_cursed": pl.Boolean,
    "magic_detected": pl.Boolean,
}


class ItemRegistry:
    def __init__(self: Self, item_templates: Dict[str, dict]):
        log.info("Initializing ItemRegistry with expanded schema")
        self.item_templates: Dict[str, dict] = item_templates

        schema_copy = ITEM_SCHEMA.copy()
        known_template_ids = list(item_templates.keys())
        if known_template_ids:
            schema_copy["template_id"] = pl.Enum(known_template_ids)
            log.debug(
                "ItemRegistry schema updated: template_id Enum",
                count=len(known_template_ids),
            )
        else:
            log.warning("ItemRegistry: No item templates provided for Enum.")

        # Add Enum categories for location_type and equipped_slot based on Literals
        schema_copy["location_type"] = pl.Enum(ItemLocation.__args__)
        schema_copy["equipped_slot"] = pl.Enum(
            list(EquipSlot.__args__)
        )  # Use list() for Enum

        try:
            # Initialize with the corrected schema
            self.items_df: pl.DataFrame = pl.DataFrame(schema=schema_copy)
        except Exception as e:
            log.error(
                "Failed to initialize ItemRegistry DataFrame",
                error=str(e),
                exc_info=True,
            )
            self.items_df = pl.DataFrame()  # Fallback

        self._next_item_id: int = 0
        log.debug("ItemRegistry initialized", templates_loaded=len(item_templates))

    def _get_next_id(self: Self) -> int:
        """Generates the next available unique item ID."""
        current_id = self._next_item_id
        self._next_item_id += 1
        if self._next_item_id > 2**64 - 1:
            log.critical("Item ID counter overflowed (UInt64 limit reached).")
            raise OverflowError("Item ID counter overflowed.")
        return current_id

    def create_item(
        self: Self,
        template_id: str,
        location: ItemLocation = "ground",
        owner_entity_id: int | None = None,
        x: int | None = None,
        y: int | None = None,
        equipped_slot: EquipSlot | None = None,  # Specific slot for 'equipped'
        quantity: int = 1,
        parent_attachment_item_id: int | None = None,  # For 'attached'
        parent_attachment_slot_id: str | None = None,  # For 'attached'
    ) -> int | None:
        """Creates a new item instance from a template. Includes attachment info."""
        template = self.item_templates.get(template_id)
        if not template:
            log.warning("Unknown item template", template_id=template_id)
            return None

        new_id = self._get_next_id()
        log_context = {"template": template_id, "item_id": new_id, "location": location}

        # --- Validation Logic ---
        if location == "ground" and (x is None or y is None):
            log.error("Ground item created without coordinates", **log_context)
            return None
        if location in ("inventory", "equipped") and owner_entity_id is None:
            log.error(f"{location} item created without owner_entity_id", **log_context)
            return None
        if location == "equipped" and equipped_slot is None:
            log.error("Equipped item created without equipped_slot", **log_context)
            return None
        if location == "attached" and (
            parent_attachment_item_id is None or parent_attachment_slot_id is None
        ):
            log.error(
                "Attached item created without parent_item_id or parent_slot_id",
                **log_context,
            )
            return None
        # --- End Validation ---

        # --- Clear conflicting location info ---
        if location != "ground":
            x, y = None, None
        if location != "equipped":
            equipped_slot = None
        if location != "attached":
            parent_attachment_item_id, parent_attachment_slot_id = None, None
        if location == "attached":
            owner_entity_id, equipped_slot = None, None
        if location in ("inventory", "equipped"):
            parent_attachment_item_id, parent_attachment_slot_id = None, None
        # --- End Clear ---

        attributes = template.get("attributes", {})
        color = template.get("color_fg", [255, 0, 255])

        template_mount_points = attributes.get("mount_points")
        template_attachable_info = attributes.get("attachable_info")

        item_data = {
            "item_id": new_id,
            "is_active": True,
            "name": template.get("name", template_id),
            "glyph": template.get("glyph", 63),
            "color_fg_r": color[0],
            "color_fg_g": color[1],
            "color_fg_b": color[2],
            "location_type": location,
            "owner_entity_id": owner_entity_id,
            "x": x,
            "y": y,
            "equipped_slot": equipped_slot,
            "quantity": max(1, quantity),
            "current_charge": attributes.get("max_charge"),
            "current_fuel": attributes.get("max_fuel"),
            "current_poison_doses": attributes.get("max_poison_doses"),
            "current_durability": attributes.get("max_durability"),
            "is_identified": False,
            "is_cursed": False,
            "magic_detected": False,
            "mount_points": (
                [
                    {**mp, "accepted_item_id": None}
                    for mp in template_mount_points
                    if isinstance(mp, dict)
                ]
                if isinstance(template_mount_points, list)
                else None
            ),
            "attachable_info": (
                template_attachable_info
                if isinstance(template_attachable_info, dict)
                else None
            ),
            "parent_attachment_item_id": parent_attachment_item_id,
            "parent_attachment_slot_id": parent_attachment_slot_id,
        }
        item_data_list = {k: [v] for k, v in item_data.items()}
        item_data_list["template_id"] = [template_id]

        try:
            new_item_df = pl.DataFrame(item_data_list)
            # --- Casting and Column Handling ---
            cast_exprs = []
            current_schema = self.items_df.schema
            for col, target_dtype in current_schema.items():
                if col in new_item_df.columns:
                    cast_exprs.append(pl.col(col).cast(target_dtype, strict=False))
            new_item_df = new_item_df.with_columns(cast_exprs)
            for col, target_dtype in current_schema.items():
                if col not in new_item_df.columns:
                    log.warning(
                        f"Column '{col}' missing in create_item data, adding null."
                    )
                    new_item_df = new_item_df.with_columns(
                        pl.lit(None, dtype=target_dtype).alias(col)
                    )
            new_item_df = new_item_df.select(current_schema.keys())
            # --- End Casting ---

            if self.items_df.height == 0:
                self.items_df = new_item_df
            else:
                self.items_df = self.items_df.vstack(new_item_df)

            log.info("Item created successfully", **log_context)
            return new_id
        except Exception as e:
            log.error(
                "Failed to create item DataFrame or vstack",
                error=str(e),
                exc_info=True,
                item_data=item_data_list,
            )
            return None

    def get_items_at(self: Self, x: int, y: int) -> pl.DataFrame:
        """Returns active items on the ground at (x, y)."""
        try:
            return self.items_df.filter(
                (pl.col("x") == x)
                & (pl.col("y") == y)
                & (pl.col("location_type") == "ground")
                & pl.col("is_active")
            )
        except Exception as e:
            log.error(
                "Error getting items at position",
                pos=(x, y),
                error=str(e),
                exc_info=True,
            )
            return pl.DataFrame(schema=ITEM_SCHEMA)  # Return empty DF

    def get_entity_inventory(self: Self, owner_entity_id: int) -> pl.DataFrame:
        """Returns active items in an entity's inventory (not equipped/attached)."""
        try:
            return self.items_df.filter(
                (pl.col("owner_entity_id") == owner_entity_id)
                & (pl.col("location_type") == "inventory")
                & pl.col("is_active")
            )
        except Exception as e:
            log.error(
                "Error getting entity inventory",
                entity_id=owner_entity_id,
                error=str(e),
                exc_info=True,
            )
            return pl.DataFrame(schema=ITEM_SCHEMA)

    def get_entity_equipped(self: Self, owner_entity_id: int) -> pl.DataFrame:
        """Returns active items equipped directly to entity slots."""
        try:
            return self.items_df.filter(
                (pl.col("owner_entity_id") == owner_entity_id)
                & (pl.col("location_type") == "equipped")
                & pl.col("is_active")
            )
        except Exception as e:
            log.error(
                "Error getting entity equipped items",
                entity_id=owner_entity_id,
                error=str(e),
                exc_info=True,
            )
            return pl.DataFrame(schema=ITEM_SCHEMA)

    def get_attached_items(self, parent_item_id: int) -> pl.DataFrame:
        """Returns active items attached to a specific parent item."""
        try:
            return self.items_df.filter(
                (pl.col("parent_attachment_item_id") == parent_item_id)
                & (pl.col("location_type") == "attached")
                & pl.col("is_active")
            )
        except Exception as e:
            log.error(
                "Error getting attached items",
                parent_id=parent_item_id,
                error=str(e),
                exc_info=True,
            )
            return pl.DataFrame(schema=ITEM_SCHEMA)  # Return empty DF

    def get_item_component(self: Self, item_id: int, component_name: str) -> Any | None:
        """Retrieves a component value for a specific active item."""
        log_context = {"item_id": item_id, "component": component_name}
        current_schema = self.items_df.schema
        if component_name not in current_schema:
            log.warning("Component does not exist in ITEM_SCHEMA", **log_context)
            raise ValueError(
                f"Item component '{component_name}' not in instance schema."
            )

        try:
            result = (
                self.items_df.lazy()
                .filter((pl.col("item_id") == item_id) & pl.col("is_active"))
                .select(component_name)
                .collect()
            )
            if result.height == 0:
                return None
            return result.item()  # Extracts the single value
        except Exception as e:
            log.error(
                "Error getting item component",
                error=str(e),
                exc_info=True,
                **log_context,
            )
            return None

    def set_item_component(
        self: Self, item_id: int, component_name: str, value: Any
    ) -> bool:
        """Sets a component value for a specific active item instance."""
        log_context = {
            "item_id": item_id,
            "component": component_name,
            "new_value": value,
        }
        current_schema = self.items_df.schema
        protected_fields = {  # Fields managed by move_item or core logic
            "item_id",
            "is_active",
            "template_id",
            "location_type",
            "owner_entity_id",
            "x",
            "y",
            "equipped_slot",
            "parent_attachment_item_id",
            "parent_attachment_slot_id",
        }
        if component_name not in current_schema:
            log.warning("Component does not exist in ITEM_SCHEMA", **log_context)
            raise ValueError(
                f"Item component '{component_name}' not in instance schema."
            )
        if component_name in protected_fields:
            log.warning(
                "Attempted to set protected/location component directly", **log_context
            )
            raise ValueError(
                f"Cannot directly set '{component_name}'. Use move_item or specific actions."
            )

        target_dtype = current_schema[component_name]
        lit_value = pl.lit(value)

        try:
            item_mask = (pl.col("item_id") == item_id) & pl.col("is_active")
            if self.items_df.filter(item_mask).height == 0:
                log.debug(
                    "Item not found or inactive, cannot set component", **log_context
                )
                return False

            self.items_df = self.items_df.with_columns(
                pl.when(item_mask)
                .then(lit_value)
                .otherwise(pl.col(component_name))
                .cast(target_dtype, strict=False)
                .alias(component_name)
            )
            return True
        except Exception as e:
            log.error(
                "Error setting item component",
                error=str(e),
                exc_info=True,
                **log_context,
            )
            return False

    def move_item(
        self: Self,
        item_id: int,
        new_location: ItemLocation,
        owner_entity_id: int | None = None,
        x: int | None = None,
        y: int | None = None,
        equipped_slot: EquipSlot | None = None,
        parent_attachment_item_id: int | None = None,
        parent_attachment_slot_id: str | None = None,
    ) -> bool:
        """Moves an active item, performing validation and state clearing."""
        log_context = {"item_id": item_id, "new_loc": new_location}

        # --- Combined Validation ---
        if new_location == "ground" and (x is None or y is None):
            log.warning("Move failed: Ground requires coords", **log_context)
            return False
        if new_location in ("inventory", "equipped") and owner_entity_id is None:
            log.warning(
                f"Move failed: {new_location} requires owner",
                **log_context,
            )
            return False
        if new_location == "equipped" and equipped_slot is None:
            log.warning("Move failed: Equipped requires slot", **log_context)
            return False
        if new_location == "attached" and (
            parent_attachment_item_id is None or parent_attachment_slot_id is None
        ):
            log.warning(
                "Move failed: Attached requires parent item/slot", **log_context
            )
            return False
        # --- End Validation ---

        item_mask = (pl.col("item_id") == item_id) & pl.col("is_active")
        if self.items_df.filter(item_mask).height == 0:
            log.warning("Move failed: Item not found or inactive", **log_context)
            return False

        # --- Prepare Updates and Clear Conflicting State ---
        final_x = x if new_location == "ground" else None
        final_y = y if new_location == "ground" else None
        final_owner = (
            owner_entity_id if new_location in ("inventory", "equipped") else None
        )
        final_equipped_slot = equipped_slot if new_location == "equipped" else None
        final_parent_item = (
            parent_attachment_item_id if new_location == "attached" else None
        )
        final_parent_slot = (
            parent_attachment_slot_id if new_location == "attached" else None
        )

        current_schema = self.items_df.schema
        updates: list[pl.Expr] = [
            pl.when(item_mask)
            .then(pl.lit(new_location))
            .otherwise(pl.col("location_type"))
            .cast(current_schema["location_type"], strict=False)
            .alias("location_type"),
            pl.when(item_mask)
            .then(pl.lit(final_owner))
            .otherwise(pl.col("owner_entity_id"))
            .cast(current_schema["owner_entity_id"], strict=False)
            .alias("owner_entity_id"),
            pl.when(item_mask)
            .then(pl.lit(final_x))
            .otherwise(pl.col("x"))
            .cast(current_schema["x"], strict=False)
            .alias("x"),
            pl.when(item_mask)
            .then(pl.lit(final_y))
            .otherwise(pl.col("y"))
            .cast(current_schema["y"], strict=False)
            .alias("y"),
            pl.when(item_mask)
            .then(pl.lit(final_equipped_slot))
            .otherwise(pl.col("equipped_slot"))
            .cast(current_schema["equipped_slot"], strict=False)
            .alias("equipped_slot"),
            pl.when(item_mask)
            .then(pl.lit(final_parent_item))
            .otherwise(pl.col("parent_attachment_item_id"))
            .cast(current_schema["parent_attachment_item_id"], strict=False)
            .alias("parent_attachment_item_id"),
            pl.when(item_mask)
            .then(pl.lit(final_parent_slot))
            .otherwise(pl.col("parent_attachment_slot_id"))
            .cast(current_schema["parent_attachment_slot_id"], strict=False)
            .alias("parent_attachment_slot_id"),
        ]

        try:
            self.items_df = self.items_df.with_columns(updates)
            log.info(
                "Item moved successfully",
                **log_context,
                owner=final_owner,
                pos=(final_x, final_y),
                slot=final_equipped_slot,
                parent=final_parent_item,
            )
            return True
        except Exception as e:
            log.error("Error moving item", error=str(e), exc_info=True, **log_context)
            return False

    def delete_item(self: Self, item_id: int) -> bool:
        """Marks an item as inactive (soft delete)."""
        log_context = {"item_id": item_id}
        try:
            item_mask = (pl.col("item_id") == item_id) & pl.col("is_active")
            was_active = self.items_df.filter(item_mask).height > 0

            if not was_active:
                log.debug("Item already inactive or does not exist", **log_context)
                return False

            self.items_df = self.items_df.with_columns(
                pl.when(item_mask)
                .then(pl.lit(False))
                .otherwise(pl.col("is_active"))
                .alias("is_active")
            )
            log.info("Item marked as inactive", **log_context)
            return True
        except Exception as e:
            log.error(
                "Error deleting item (marking inactive)",
                error=str(e),
                exc_info=True,
                **log_context,
            )
            return False

    def compact_registry(self: Self) -> None:
        """Permanently removes inactive items."""
        log.info("Compacting item registry...")
        try:
            initial_count = self.items_df.height
            self.items_df = self.items_df.filter(pl.col("is_active"))
            final_count = self.items_df.height
            removed_count = initial_count - final_count
            log.info(
                "Item registry compacted",
                initial=initial_count,
                final=final_count,
                removed=removed_count,
            )
        except Exception as e:
            log.error("Error compacting item registry", error=str(e), exc_info=True)

    # --- Utility Methods ---
    def get_template(self: Self, template_id: str) -> dict | None:
        """Safely retrieve an item template definition."""
        return self.item_templates.get(template_id)

    def get_item_template_id(self: Self, item_id: int) -> str | None:
        """Retrieve the template_id for a given item_id."""
        template_id_val = self.get_item_component(item_id, "template_id")
        return cast(str, template_id_val) if template_id_val else None

    def get_item_static_attribute(
        self: Self, item_id: int, attribute_name: str, default: Any = None
    ) -> Any:
        """Convenience method to get a static attribute from an item's template."""
        template_id = self.get_item_template_id(item_id)
        if template_id:
            template = self.get_template(template_id)
            if template:
                return template.get("attributes", {}).get(attribute_name, default)
        return default

    # --- New Flag Helpers ---
    def get_item_flags(self: Self, item_id: int) -> list[str]:
        """Return the list of flags defined for an item's template."""
        template_id = self.get_item_template_id(item_id)
        if template_id:
            template = self.get_template(template_id)
            if template:
                flags = template.get("flags", [])
                if isinstance(flags, list):
                    # Ensure everything is string for comparisons
                    return [str(f) for f in flags]
        return []

    def item_has_flag(self: Self, item_id: int, flag: str) -> bool:
        """Check whether the item's template declares a specific flag."""
        return flag in self.get_item_flags(item_id)
