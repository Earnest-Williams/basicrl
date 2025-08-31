# game/systems/equipment_system.py

"""
Handles the logic for equipping, unequipping, attaching, and detaching items,
considering entity body plans and item mount points.
"""
# Ensure Union is imported from typing
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union, cast

import polars as pl
import structlog

from game.entities.components import Inventory
from game.effects.executor import execute_effect

# Import specific types needed from other modules
# Ensure these imports work correctly based on your project structure
try:
    from game.entities.registry import EntityRegistry
    from game.items.registry import BodySlotType, EquipSlot, ItemLocation, ItemRegistry
except ImportError as e:
    # Define logger early for import errors
    # Note: Basic config might be needed if structlog isn't fully configured yet
    import logging

    logging.basicConfig(level=logging.ERROR)
    log_fallback = logging.getLogger(__name__)
    log_fallback.error(
        f"CRITICAL: Failed to import registry types in equipment_system: {e}"
    )
    # Define dummy types to prevent NameErrors during parsing, but expect runtime failures
    EquipSlot = Any
    ItemLocation = Any
    BodySlotType = Any
    EntityRegistry = Any
    ItemRegistry = Any
    # Fallback logger if full structlog isn't ready
    log = log_fallback


if TYPE_CHECKING:
    from game.game_state import GameState

# Setup logger properly if structlog is available
try:
    log = structlog.get_logger(__name__)
except NameError:  # Handles case where log_fallback was used above
    pass


# --- Helper Functions ---


def get_item_template(item_id: int, gs: "GameState") -> Dict | None:
    """Helper: Safely get the template data for an item."""
    # Check if item_registry exists on gs
    if not hasattr(gs, "item_registry") or gs.item_registry is None:
        log.error("GameState missing item_registry", method="get_item_template")
        return None
    template_id = gs.item_registry.get_item_template_id(item_id)
    if template_id:
        return gs.item_registry.get_template(template_id)
    # Avoid logging warning if item_id itself was None or invalid
    if item_id is not None:
        log.debug(
            "Could not find template for item", item_id=item_id
        )  # Changed from warning
    return None


def get_item_attribute(
    item_id: int, attribute_name: str, gs: "GameState", default: Any = None
) -> Any:
    """Helper: Safely get an attribute from an item's template data."""
    template = get_item_template(item_id, gs)
    if template:
        return template.get("attributes", {}).get(attribute_name, default)
    return default


def get_general_slot_type(specific_slot: "EquipSlot") -> str | None:
    """Maps a specific EquipSlot (e.g., finger_3) to a general BodySlotType (e.g., finger)."""
    # Ensure specific_slot is a string before calling startswith
    if not isinstance(specific_slot, str):
        # Allow None gracefully
        if specific_slot is None:
            return None
        log.warning(
            "Invalid specific_slot type passed to get_general_slot_type",
            type=type(specific_slot),
        )
        return None

    if specific_slot.startswith("finger_"):
        return "finger"
    if specific_slot.startswith("neck_"):
        return "neck"
    if specific_slot in ["main_hand", "off_hand"]:
        return "hand"
    if specific_slot == "wrists":
        return "wrist"
    # Direct mappings based on BodySlotType keys - requires BodySlotType to be defined/imported
    try:
        if specific_slot in BodySlotType.__args__:
            return specific_slot
    except AttributeError:  # Handle if BodySlotType wasn't imported correctly
        log.error("BodySlotType type definition not available for validation")
        # Cannot safely map, return None
        return None

    log.warning(
        "Could not map specific slot to general type", specific_slot=specific_slot
    )
    return None


def get_slots_occupied_by_general_type(
    entity_id: int, general_slot_type: str, gs: "GameState"
) -> List["EquipSlot"]:
    """Helper: Gets a list of specific slots occupied by items of a general type."""
    occupied_slots = []
    # Ensure registries exist
    if (
        not hasattr(gs, "entity_registry")
        or gs.entity_registry is None
        or not hasattr(gs, "item_registry")
        or gs.item_registry is None
    ):
        log.error(
            "GameState missing entity_registry or item_registry",
            method="get_slots_occupied_by_general_type",
        )
        return []

    entity_reg = gs.entity_registry
    item_reg = gs.item_registry
    equipped_ids = entity_reg.get_equipped_ids(entity_id)
    if not equipped_ids:
        return []

    try:
        if equipped_ids:
            equipped_items_df = item_reg.items_df.filter(
                pl.col("item_id").is_in(equipped_ids)
                & (pl.col("location_type") == "equipped")
            )
            # Use select().to_series() for potentially faster extraction than iter_rows
            equipped_slots_series = (
                equipped_items_df.select("equipped_slot").to_series().drop_nulls()
            )

            for specific_slot in equipped_slots_series:
                if (
                    specific_slot
                ):  # Check if not None after drop_nulls (though redundant)
                    current_general_type = get_general_slot_type(
                        cast(str, specific_slot)
                    )  # Cast for safety
                    if current_general_type == general_slot_type:
                        # Ensure the slot is a valid EquipSlot before appending
                        try:
                            if (
                                "EquipSlot" in globals()
                                and EquipSlot is not Any
                                and specific_slot in EquipSlot.__args__
                            ):
                                occupied_slots.append(cast(EquipSlot, specific_slot))
                            # If EquipSlot wasn't loaded, we might have appended string earlier, allow it but log
                            elif "EquipSlot" not in globals() or EquipSlot is Any:
                                occupied_slots.append(
                                    specific_slot
                                )  # Assume it's valid if type unknown
                            else:
                                log.warning(
                                    "Occupied slot is not a valid EquipSlot literal",
                                    slot=specific_slot,
                                )
                        except AttributeError:
                            log.error(
                                "EquipSlot type definition not available for validation"
                            )
                            occupied_slots.append(specific_slot)  # Append anyway

        return occupied_slots
    except Exception as e:
        log.error(
            "Error fetching/processing equipped slots",
            entity_id=entity_id,
            error=str(e),
            exc_info=True,
        )
        return []


# --- CORRECTED TYPE HINT using Union ---
def find_first_available_slot(
    entity_id: int, item_id: int, gs: "GameState"
) -> Union[EquipSlot, None]:
    # --- End Correction ---
    """Helper: Find the first specific, unoccupied EquipSlot compatible with the item."""
    item_reg = gs.item_registry
    entity_reg = gs.entity_registry
    if not item_reg or not entity_reg:
        log.error("Registries unavailable", sys="equipment")
        return None

    template = get_item_template(item_id, gs)
    if not template:
        return None

    compatible_slots_specific = get_item_attribute(
        item_id, "compatible_equip_slots", gs
    )  # Option A
    primary_equip_slot = template.get("equip_slot")  # Option C (primary definition)
    general_slot_type = get_item_attribute(
        item_id, "general_equip_type", gs
    )  # Option B

    target_general_type = None
    possible_specific_slots: List[EquipSlot] = []

    # Determine valid EquipSlot literals dynamically if possible
    try:
        valid_equip_slots = (
            set(EquipSlot.__args__)
            if "EquipSlot" in globals() and EquipSlot is not Any
            else set()
        )
    except AttributeError:
        log.error("EquipSlot type definition not available for validation")
        valid_equip_slots = set()  # Cannot validate

    if isinstance(compatible_slots_specific, list) and compatible_slots_specific:
        possible_specific_slots = [
            s for s in compatible_slots_specific if s in valid_equip_slots
        ]
        if possible_specific_slots:
            target_general_type = get_general_slot_type(possible_specific_slots[0])
        else:
            log.warning(
                "Item compatible_equip_slots list contains no valid slots",
                item_id=item_id,
            )
            return None
    elif primary_equip_slot and primary_equip_slot in valid_equip_slots:
        target_general_type = get_general_slot_type(primary_equip_slot)
        possible_specific_slots = [
            s
            for s in valid_equip_slots
            if get_general_slot_type(s) == target_general_type
        ]
    elif general_slot_type:
        target_general_type = general_slot_type
        possible_specific_slots = [
            s
            for s in valid_equip_slots
            if get_general_slot_type(s) == target_general_type
        ]
    else:
        log.warning(
            "Item template lacks equip slot definition",
            item_id=item_id,
            template_id=template.get("name", "?"),
        )
        return None

    if not target_general_type or not possible_specific_slots:
        log.warning(
            "Could not determine target slot type or possible specific slots",
            item_id=item_id,
            general=target_general_type,
        )
        return None

    # Check body plan vs occupied count
    body_plan = entity_reg.get_body_plan(entity_id)
    if not body_plan:
        log.error("Entity has no body plan", entity_id=entity_id)
        return None
    available_count = body_plan.get(target_general_type, 0)
    occupied_slots = get_slots_occupied_by_general_type(
        entity_id, target_general_type, gs
    )
    current_count = len(occupied_slots)

    if current_count >= available_count:
        log.debug(
            "No free slots of general type",
            entity_id=entity_id,
            type=target_general_type,
            equipped=current_count,
            available=available_count,
        )
        return None

    # Find first free specific slot from the possible list
    occupied_set = set(occupied_slots)
    for slot in possible_specific_slots:
        if slot not in occupied_set:
            # Final check: ensure the found slot literal is valid
            if slot in valid_equip_slots:
                return slot
            else:  # Should not happen if possible_specific_slots was filtered correctly
                log.error(
                    "Internal Error: Found slot not in valid EquipSlot literals",
                    slot=slot,
                )
                return None

    log.debug(
        "Could not find specific unoccupied slot despite count mismatch",
        type=target_general_type,
        occupied=occupied_set,
        possible=possible_specific_slots,
    )
    return None


def get_item_mount_points(item_id: int, gs: "GameState") -> List[Dict] | None:
    """Helper: Get mount_points list for an item, ensuring validity."""
    if not hasattr(gs, "item_registry") or gs.item_registry is None:
        return None
    mounts = gs.item_registry.get_item_component(item_id, "mount_points")
    if isinstance(mounts, list) and all(isinstance(mp, dict) for mp in mounts):
        return mounts
    elif mounts is not None:
        log.warning(
            "mount_points component data is not list[dict]",
            item_id=item_id,
            data=mounts,
        )
    return None  # Return None if missing or invalid format


def get_attachable_info(item_id: int, gs: "GameState") -> Dict | None:
    """Helper: Get attachable_info dict for an item."""
    if not hasattr(gs, "item_registry") or gs.item_registry is None:
        return None
    info = gs.item_registry.get_item_component(item_id, "attachable_info")
    return info if isinstance(info, dict) else None


# --- Passive Effect Helpers ---


def apply_passive_effects(item_id: int, entity_id: int, gs: "GameState") -> None:
    """Applies passive effects defined on an item's template."""
    if not hasattr(gs, "item_registry") or gs.item_registry is None:
        log.error("GameState missing item_registry", method="apply_passive_effects")
        return
    template = get_item_template(item_id, gs)
    if not template:
        return
    effect_ids = template.get("effects", {}).get("passive", [])
    if not effect_ids:
        return
    for effect_id in effect_ids:
        if not isinstance(effect_id, str):
            continue
        context = {
            "game_state": gs,
            "source_entity_id": entity_id,
            "target_entity_id": entity_id,
            "item_instance_id": item_id,
            "rng": getattr(gs, "rng_instance", None),
        }
        try:
            execute_effect(effect_id, context)
        except Exception as e:
            log.error(
                "Failed to apply passive effect",
                item_id=item_id,
                effect_id=effect_id,
                error=str(e),
            )


def remove_passive_effects(item_id: int, entity_id: int, gs: "GameState") -> None:
    """Removes passive effects defined on an item's template."""
    if not hasattr(gs, "item_registry") or gs.item_registry is None:
        log.error("GameState missing item_registry", method="remove_passive_effects")
        return
    template = get_item_template(item_id, gs)
    if not template:
        return
    effect_ids = template.get("effects", {}).get("passive_remove", [])
    if not effect_ids:
        return
    for effect_id in effect_ids:
        if not isinstance(effect_id, str):
            continue
        context = {
            "game_state": gs,
            "source_entity_id": entity_id,
            "target_entity_id": entity_id,
            "item_instance_id": item_id,
            "rng": getattr(gs, "rng_instance", None),
        }
        try:
            execute_effect(effect_id, context)
        except Exception as e:
            log.error(
                "Failed to remove passive effect",
                item_id=item_id,
                effect_id=effect_id,
                error=str(e),
            )


# --- Main Equipment Actions ---


def can_equip(
    entity_id: int, item_id: int, gs: "GameState"
) -> Tuple[bool, str, Union[EquipSlot, None]]:
    """Checks if entity can equip item directly. Returns (can_equip, reason, target_slot)."""
    log.debug("Checking can_equip", entity_id=entity_id, item_id=item_id)
    # Ensure registries are available
    if (
        not hasattr(gs, "item_registry")
        or gs.item_registry is None
        or not hasattr(gs, "entity_registry")
        or gs.entity_registry is None
    ):
        return False, "Internal Error: Registry missing", None
    item_reg = gs.item_registry

    # Check if item exists and is active
    item_state = (
        item_reg.items_df.filter(pl.col("item_id") == item_id)
        .select(["is_active", "location_type"])
        .row(0, named=True, default=None)
    )
    if not item_state or not item_state.get("is_active"):
        return False, "Item does not exist or is inactive", None

    # Check if item is equippable at all
    template = get_item_template(item_id, gs)
    if not template:
        return False, "Invalid item template", None
    flags = template.get("flags", [])
    is_defined_equippable = (
        template.get("equip_slot")
        or get_item_attribute(item_id, "compatible_equip_slots", gs)
        or get_item_attribute(item_id, "general_equip_type", gs)
    )
    if "EQUIPPABLE" not in flags and not is_defined_equippable:
        return False, "Item not equippable", None

    # Check if item is already equipped or attached elsewhere
    loc_type = item_state.get("location_type")
    if loc_type not in ["inventory", "ground"]:
        return False, f"Item is currently {loc_type}", None

    # Find available slot
    target_slot = find_first_available_slot(entity_id, item_id, gs)
    if target_slot:
        return True, "Slot available", target_slot
    else:
        # Check if the failure was due to lack of compatible slot type in body_plan
        general_type = get_item_attribute(item_id, "general_equip_type", gs) or (
            get_general_slot_type(template["equip_slot"])
            if template.get("equip_slot")
            else None
        )
        if (
            general_type
            and gs.entity_registry.get_body_plan(entity_id).get(general_type, 0) == 0
        ):
            return False, f"Body has no '{general_type}' slots", None
        return False, "No compatible/free slot", None


def equip_item(entity_id: int, item_id: int, gs: "GameState") -> bool:
    """Equips item to the first available valid slot on the entity."""
    if (
        not hasattr(gs, "entity_registry")
        or gs.entity_registry is None
        or not hasattr(gs, "item_registry")
        or gs.item_registry is None
    ):
        log.error("GameState missing registries", method="equip_item")
        return False
    entity_reg = gs.entity_registry
    item_reg = gs.item_registry

    can, reason, target_slot = can_equip(entity_id, item_id, gs)
    if not can or target_slot is None:
        log.debug(f"Cannot equip item {item_id} on entity {entity_id}: {reason}")
        if entity_id == gs.player_id:
            gs.add_message(f"Cannot equip that: {reason}")
        return False  # Turn not consumed

    log.info("Equipping item", entity_id=entity_id, item_id=item_id, slot=target_slot)

    current_ids = entity_reg.get_equipped_ids(entity_id)
    if current_ids is None:
        log.error("Failed to get equipped_ids list", entity_id=entity_id)
        return False
    if item_id in current_ids:
        log.warning(
            "Item already in equipped list? Aborting equip.",
            item_id=item_id,
            entity_id=entity_id,
        )
        # Maybe unequip first if logic requires it? For now, just prevent re-equip.
        gs.add_message("That item seems to already be equipped.", (255, 150, 0))
        return False  # Prevent duplicate equip
    new_ids = current_ids + [item_id]

    if not entity_reg.set_equipped_ids(entity_id, new_ids):
        log.error("Failed to set equipped_ids list", entity_id=entity_id)
        return False

    success = item_reg.move_item(
        item_id=item_id,
        new_location="equipped",
        owner_entity_id=entity_id,
        equipped_slot=target_slot,
    )

    if not success:
        log.error(
            "Failed to move item to equipped state in ItemRegistry", item_id=item_id
        )
        try:
            entity_reg.set_equipped_ids(entity_id, current_ids)  # Attempt rollback
        except Exception as rollback_err:
            log.critical(
                "Rollback equip failed!", entity_id=entity_id, error=rollback_err
            )
        return False

    item_name = item_reg.get_item_component(item_id, "name") or "item"
    if entity_id == gs.player_id:
        gs.add_message(f"You equip the {item_name} ({target_slot}).")
    apply_passive_effects(item_id, entity_id, gs)

    return True


def can_unequip(entity_id: int, item_id: int, gs: "GameState") -> Tuple[bool, str]:
    """Checks if entity can unequip item. Checks for attached items and inventory space."""
    if (
        not hasattr(gs, "entity_registry")
        or gs.entity_registry is None
        or not hasattr(gs, "item_registry")
        or gs.item_registry is None
    ):
        log.error("GameState missing registries", method="can_unequip")
        return False, "Internal Error"
    item_reg = gs.item_registry
    entity_reg = gs.entity_registry

    # Check if item exists and is active first
    item_data = (
        item_reg.items_df.filter(pl.col("item_id") == item_id)
        .select(["is_active", "location_type"])
        .row(0, named=True, default=None)
    )
    if not item_data or not item_data.get("is_active"):
        return False, "Item does not exist"

    loc_type = item_data.get("location_type")
    if loc_type != "equipped":
        return False, "Item not equipped"

    # Check ownership via equipped_ids list (more robust than checking item.owner_entity_id)
    equipped_ids = entity_reg.get_equipped_ids(entity_id)
    if equipped_ids is None or item_id not in equipped_ids:
        log.warning(
            "Item location is 'equipped' but not in entity's equipped_ids list!",
            item_id=item_id,
            entity_id=entity_id,
        )
        return False, "Item not equipped by you"  # Or Internal Error?

    mount_points = get_item_mount_points(item_id, gs)
    if mount_points:
        for mp in mount_points:
            if mp.get("accepted_item_id") is not None:
                # Fetch name safely
                attached_item_name = "something"
                try:
                    attached_item_name = (
                        item_reg.get_item_component(mp["accepted_item_id"], "name")
                        or "something"
                    )
                except:
                    pass  # Ignore if attached item doesn't exist
                return False, f"'{attached_item_name}' attached"

    inv_capacity = entity_reg.get_entity_component(entity_id, "inventory_capacity")
    if inv_capacity is not None:
        current_inventory = item_reg.get_entity_inventory(entity_id)
        inventory = Inventory(
            capacity=inv_capacity,
            items=current_inventory.select("item_id").to_series().to_list(),
        )
        if len(inventory.items) >= inventory.capacity:
            return False, "Inventory full"

    return True, "Can unequip"


def unequip_item(entity_id: int, item_id: int, gs: "GameState") -> bool:
    """Unequips item from entity, moving to inventory."""
    if (
        not hasattr(gs, "entity_registry")
        or gs.entity_registry is None
        or not hasattr(gs, "item_registry")
        or gs.item_registry is None
    ):
        log.error("GameState missing registries", method="unequip_item")
        return False
    entity_reg = gs.entity_registry
    item_reg = gs.item_registry

    can, reason = can_unequip(entity_id, item_id, gs)
    if not can:
        log.debug(f"Cannot unequip item {item_id} from entity {entity_id}: {reason}")
        if entity_id == gs.player_id:
            gs.add_message(f"Cannot unequip: {reason}.")
        return False

    log.info("Unequipping item", entity_id=entity_id, item_id=item_id)

    current_ids = entity_reg.get_equipped_ids(entity_id)
    if current_ids is None:
        log.error("Failed to get equipped_ids list during unequip", entity_id=entity_id)
        return False
    try:
        new_ids = [i for i in current_ids if i != item_id]
    except TypeError:
        log.error(
            "equipped_item_ids was not a list during unequip", entity_id=entity_id
        )
        return False

    if not entity_reg.set_equipped_ids(entity_id, new_ids):
        log.error("Failed to set equipped_ids list during unequip", entity_id=entity_id)
        return False

    success = item_reg.move_item(
        item_id=item_id, new_location="inventory", owner_entity_id=entity_id
    )

    if not success:
        log.error(
            "Failed to move item to inventory state in ItemRegistry during unequip",
            item_id=item_id,
        )
        try:
            entity_reg.set_equipped_ids(entity_id, current_ids)  # Rollback attempt
        except Exception as rb_err:
            log.critical("Rollback unequip failed!", entity_id=entity_id, error=rb_err)
        return False

    item_name = item_reg.get_item_component(item_id, "name") or "item"
    if entity_id == gs.player_id:
        gs.add_message(f"You unequip the {item_name}.")
    remove_passive_effects(item_id, entity_id, gs)

    return True


# --- Attachment Functions ---


def can_attach(
    entity_id: int, item_to_attach_id: int, target_host_item_id: int, gs: "GameState"
) -> Tuple[bool, str, str | None]:
    """Checks if item can be attached to host item. Returns (can_attach, reason, mount_slot_id)."""
    if (
        not hasattr(gs, "entity_registry")
        or gs.entity_registry is None
        or not hasattr(gs, "item_registry")
        or gs.item_registry is None
    ):
        log.error("GameState missing registries", method="can_attach")
        return False, "Internal Error", None
    item_reg = gs.item_registry
    entity_reg = gs.entity_registry

    # Check items exist and are active
    attach_item_data = (
        item_reg.items_df.filter(pl.col("item_id") == item_to_attach_id)
        .select(["is_active", "location_type", "owner_entity_id"])
        .row(0, named=True, default=None)
    )
    host_item_data = (
        item_reg.items_df.filter(pl.col("item_id") == target_host_item_id)
        .select(["is_active", "location_type", "owner_entity_id"])
        .row(0, named=True, default=None)
    )

    if not attach_item_data or not attach_item_data.get("is_active"):
        return False, "Item to attach does not exist", None
    if not host_item_data or not host_item_data.get("is_active"):
        return False, "Host item does not exist", None

    # 1. Check item_to_attach is in entity's inventory
    if (
        attach_item_data.get("location_type") != "inventory"
        or attach_item_data.get("owner_entity_id") != entity_id
    ):
        return False, "Item not in inventory", None

    # 2. Check target_host_item is equipped by the entity
    if (
        host_item_data.get("location_type") != "equipped"
        or host_item_data.get("owner_entity_id") != entity_id
    ):
        return False, "Host item not equipped", None

    # 3. Get attachable_info from item_to_attach
    attach_info = get_attachable_info(item_to_attach_id, gs)
    if not attach_info:
        return False, "Item not attachable", None
    compatible_mount_types = attach_info.get("compatible_mount_types")
    if not compatible_mount_types or not isinstance(compatible_mount_types, list):
        return False, "Item has no compatible mount types defined", None

    # 4. Get mount_points from target_host_item
    mount_points = get_item_mount_points(target_host_item_id, gs)
    if not mount_points:
        return False, "Host item has no mount points", None

    # 5. Iterate through mount_points on host and find compatible + free slot
    for mp in mount_points:
        mp_id = mp.get("id")
        is_free = mp.get("accepted_item_id") is None
        if mp_id and is_free and mp_id in compatible_mount_types:
            # TODO: Add flag checks if needed
            return True, "Compatible slot found", mp_id

    return False, "No compatible/free mount points", None


def attach_item(
    entity_id: int, item_to_attach_id: int, target_host_item_id: int, gs: "GameState"
) -> bool:
    """Attaches an item to a host item."""
    if not hasattr(gs, "item_registry") or gs.item_registry is None:
        log.error("GameState missing item_registry", method="attach_item")
        return False
    item_reg = gs.item_registry

    can, reason, mount_slot_id = can_attach(
        entity_id, item_to_attach_id, target_host_item_id, gs
    )
    if not can or mount_slot_id is None:
        log.debug(
            f"Cannot attach item {item_to_attach_id} to {target_host_item_id}: {reason}"
        )
        if entity_id == gs.player_id:
            gs.add_message(f"Cannot attach: {reason}.")
        return False

    log.info(
        "Attaching item",
        entity_id=entity_id,
        child_item=item_to_attach_id,
        parent_item=target_host_item_id,
        mount_id=mount_slot_id,
    )

    mount_points_list = get_item_mount_points(target_host_item_id, gs)
    if mount_points_list is None:
        log.error(
            "Host item mount points disappeared between check and action",
            host_item=target_host_item_id,
        )
        return False

    found_slot = False
    modified_mount_points = []
    for mp in mount_points_list:
        mp_copy = mp.copy()
        if (
            mp_copy.get("id") == mount_slot_id
            and mp_copy.get("accepted_item_id") is None
        ):
            mp_copy["accepted_item_id"] = item_to_attach_id
            found_slot = True
        modified_mount_points.append(mp_copy)

    if not found_slot:
        log.error(
            "Mount slot became occupied between check and action",
            host_item=target_host_item_id,
            slot_id=mount_slot_id,
        )
        return False

    if not item_reg.set_item_component(
        target_host_item_id, "mount_points", modified_mount_points
    ):
        log.error(
            "Failed to update host item mount points", host_item=target_host_item_id
        )
        return False

    success = item_reg.move_item(
        item_id=item_to_attach_id,
        new_location="attached",
        parent_attachment_item_id=target_host_item_id,
        parent_attachment_slot_id=mount_slot_id,
    )

    if not success:
        log.error("Failed to move attached item state", child_item=item_to_attach_id)
        log.critical(
            "CRITICAL: Failed to move attached item after host mount point update! State inconsistent.",
            child_item=item_to_attach_id,
            host_item=target_host_item_id,
        )
        # Rollback mount point change? Difficult.
        return False

    child_name = item_reg.get_item_component(item_to_attach_id, "name") or "item"
    host_name = item_reg.get_item_component(target_host_item_id, "name") or "item"
    if entity_id == gs.player_id:
        gs.add_message(f"You attach the {child_name} to the {host_name}.")
    apply_passive_effects(item_to_attach_id, entity_id, gs)

    return True


def can_detach(
    entity_id: int, item_to_detach_id: int, gs: "GameState"
) -> Tuple[bool, str]:
    """Checks if item can be detached by the entity."""
    if (
        not hasattr(gs, "entity_registry")
        or gs.entity_registry is None
        or not hasattr(gs, "item_registry")
        or gs.item_registry is None
    ):
        log.error("GameState missing registries", method="can_detach")
        return False, "Internal Error"
    item_reg = gs.item_registry
    entity_reg = gs.entity_registry

    item_info = (
        item_reg.items_df.filter(pl.col("item_id") == item_to_detach_id)
        .select(["is_active", "location_type", "parent_attachment_item_id"])
        .row(0, named=True, default=None)
    )

    if not item_info or not item_info.get("is_active"):
        return False, "Item does not exist"
    if item_info.get("location_type") != "attached":
        return False, "Item not attached"

    parent_item_id = item_info.get("parent_attachment_item_id")
    if parent_item_id is None:
        log.error("Attached item missing parent ID", item_id=item_to_detach_id)
        return False, "Internal error (missing parent)"

    # Check ownership chain
    parent_owner_id = item_reg.get_item_component(parent_item_id, "owner_entity_id")
    if parent_owner_id != entity_id:
        return False, "You don't own the item it's attached to"

    # Check inventory space
    inv_capacity = entity_reg.get_entity_component(entity_id, "inventory_capacity")
    if inv_capacity is not None:
        current_inventory = item_reg.get_entity_inventory(entity_id)
        inventory = Inventory(
            capacity=inv_capacity,
            items=current_inventory.select("item_id").to_series().to_list(),
        )
        if len(inventory.items) >= inventory.capacity:
            return False, "Inventory full"

    return True, "Can detach"


def detach_item(entity_id: int, item_to_detach_id: int, gs: "GameState") -> bool:
    """Detaches an item, moving it to entity's inventory."""
    if not hasattr(gs, "item_registry") or gs.item_registry is None:
        log.error("GameState missing item_registry", method="detach_item")
        return False
    item_reg = gs.item_registry

    can, reason = can_detach(entity_id, item_to_detach_id, gs)
    if not can:
        log.debug(f"Cannot detach item {item_to_detach_id}: {reason}")
        if entity_id == gs.player_id:
            gs.add_message(f"Cannot detach: {reason}.")
        return False

    log.info("Detaching item", entity_id=entity_id, item_id=item_to_detach_id)

    detach_info = (
        item_reg.items_df.filter(pl.col("item_id") == item_to_detach_id)
        .select(["parent_attachment_item_id", "parent_attachment_slot_id"])
        .row(0, named=True, default=None)
    )

    if not detach_info:
        log.error("Failed to get detach info", item_id=item_to_detach_id)
        return False
    parent_item_id = detach_info.get("parent_attachment_item_id")
    parent_slot_id = detach_info.get("parent_attachment_slot_id")
    if parent_item_id is None or parent_slot_id is None:
        log.error("Detach failed: Item missing parent info.", item_id=item_to_detach_id)
        return False

    mount_points_list = get_item_mount_points(parent_item_id, gs)
    if mount_points_list is None:
        log.error(
            "Detach failed: Parent item has no valid mount points.",
            parent_item_id=parent_item_id,
        )
        return False

    found_slot = False
    modified_mount_points = []
    for mp in mount_points_list:
        mp_copy = mp.copy()
        if (
            mp_copy.get("id") == parent_slot_id
            and mp_copy.get("accepted_item_id") == item_to_detach_id
        ):
            mp_copy["accepted_item_id"] = None
            found_slot = True
        modified_mount_points.append(mp_copy)

    if not found_slot:
        log.error(
            "Detach failed: Could not find item in parent's mount slot.",
            item_id=item_to_detach_id,
            parent_item_id=parent_item_id,
            slot_id=parent_slot_id,
        )
        return False

    if not item_reg.set_item_component(
        parent_item_id, "mount_points", modified_mount_points
    ):
        log.error(
            "Detach failed: Could not update parent item mount points",
            parent_item_id=parent_item_id,
        )
        return False

    success = item_reg.move_item(
        item_id=item_to_detach_id, new_location="inventory", owner_entity_id=entity_id
    )

    if not success:
        log.error(
            "Detach failed: Could not move item to inventory", item_id=item_to_detach_id
        )
        log.critical(
            "CRITICAL: Failed to move detached item after clearing parent mount point! State inconsistent.",
            item_id=item_to_detach_id,
            parent_item_id=parent_item_id,
        )
        # Rollback is hard here
        return False

    item_name = item_reg.get_item_component(item_to_detach_id, "name") or "item"
    parent_name = item_reg.get_item_component(parent_item_id, "name") or "item"
    if entity_id == gs.player_id:
        gs.add_message(f"You detach the {item_name} from the {parent_name}.")
    remove_passive_effects(item_to_detach_id, entity_id, gs)

    return True


def handle_limb_loss(entity_id: int, lost_slot_type: str, gs: "GameState"):
    """Handles unequipping items when a body slot type count decreases."""
    log.warning("handle_limb_loss needs implementation")
    # 1. Get body_plan, check new count for lost_slot_type.
    # 2. Get list of specific slots occupied by items of lost_slot_type.
    # 3. If len(occupied) > new_count:
    # 4.   Determine which specific slots/items are lost (e.g., highest index finger).
    # 5.   For each lost item: call unequip_item(entity_id, item_id_to_remove, gs).
    #      If unequip fails due to inventory full, attempt to drop instead?
    pass
