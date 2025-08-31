# engine/action_handler.py
"""
Handles processing of player actions, validating them against game rules,
and triggering appropriate game state changes or effect executions. Includes falling
and melee attack initiation.
"""
from typing import Any, Dict, Tuple  # Added Tuple

import structlog

from game.effects.executor import execute_effect
from game.entities.registry import EntityRegistry
from game.entities.components import Position

# Use absolute imports for game modules
from game.game_state import GameState
from game.items.registry import ItemRegistry
from game.world.game_map import GameMap
from game.systems import movement_system
from game.systems.death_system import handle_entity_death

# Import equipment system safely
try:
    from game.systems import equipment_system
except ImportError:
    _log_early = structlog.get_logger()
    _log_early.error("CRITICAL: Failed to import game.systems.equipment_system.")

    class DummyEquipmentSystem:
        def __getattr__(self, name):
            def method(*args, **kwargs):
                log = (
                    structlog.get_logger()
                )  # Logger inside method to avoid early init issues
                log.error(f"Equipment system not loaded. Call to {name} failed.")
                return False

            return method

    equipment_system = DummyEquipmentSystem()

# --- Import the combat system safely ---
try:
    from game.systems import combat_system
except ImportError:
    _log_early = structlog.get_logger()
    _log_early.error("CRITICAL: Failed to import game.systems.combat_system.")

    # Define dummy function
    class DummyCombatSystem:
        def handle_melee_attack(self, *args, **kwargs):
            log = structlog.get_logger()  # Logger inside method
            log.error("Combat system not loaded. Call to handle_melee_attack failed.")
            # If combat fails, the action (attack attempt) effectively does nothing
            # but should still consume the turn. We return False here to indicate
            # the *system* failed, but process_player_action will handle turn cost.
            # Returning False here aligns better with other system failures.
            return False

    combat_system = DummyCombatSystem()
# --- End Import ---


log = structlog.get_logger(__name__)  # Define module logger after potential early logs


# --- Fall Parameters ---
FALL_DAMAGE_THRESHOLD: int = 3  # Units (e.g., 1.5 meters)
FALL_DAMAGE_PER_UNIT_HEIGHT: float = 2.0  # Damage per unit height beyond threshold
MAX_FALL_DEPTH: int = 20  # Max tiles to check downwards


# --- Fall Handling Helper ---
def _handle_fall(
    entity_id: int,
    start_pos_h: Tuple[int, int, int],  # x, y, h
    fall_trigger_pos: Tuple[int, int, int],  # nx, ny, nh
    gs: GameState,
    max_step: int,  # Pass max_traversable_step
) -> bool:
    """
    Handles the logic when an entity moves off an edge too steep to step down.
    Returns True if the fall occurred (consuming a turn), False otherwise.
    """
    start_x, start_y, start_h = start_pos_h
    trigger_x, trigger_y, trigger_h = fall_trigger_pos
    entity_reg: EntityRegistry = gs.entity_registry  # Add type hint
    game_map: GameMap = gs.game_map  # Add type hint

    log.debug(
        "Entity falling",
        entity_id=entity_id,
        from_pos=(start_x, start_y, start_h),
        trigger_pos=(trigger_x, trigger_y, trigger_h),
    )
    gs.add_message("You fall!", (255, 100, 0))  # Simple message for now

    landing_x, landing_y, landing_h = trigger_x, trigger_y, trigger_h
    current_check_y = trigger_y

    # Trace downwards to find landing spot
    for depth in range(MAX_FALL_DEPTH):
        next_y = current_check_y + 1

        # Check map bounds
        if not game_map.in_bounds(landing_x, next_y):
            log.debug(
                "Fall hit bottom of map", entity_id=entity_id, last_y=current_check_y
            )
            landing_y = current_check_y
            landing_h = int(game_map.height_map[landing_y, landing_x])
            break

        # Get height of the tile below the current check position
        tile_below_h = int(game_map.height_map[next_y, landing_x])
        # Check step down from current potential landing spot
        step_down_delta = abs(landing_h - tile_below_h)

        # Check if the step down from the *current* landing spot is stable
        if step_down_delta <= max_step:
            # Found stable ground below *this* position
            log.debug(
                "Fall landing spot found",
                entity_id=entity_id,
                pos=(landing_x, landing_y, landing_h),
                depth=depth,
            )
            break  # Land at current landing_x, landing_y
        else:
            # Continue falling, update the 'current landing spot' for the next iteration's check
            current_check_y = next_y  # Move check position down first
            landing_y = current_check_y
            landing_h = tile_below_h

        # Check if we hit max depth after potentially updating position
        if depth == MAX_FALL_DEPTH - 1:
            log.warning(
                "Max fall depth reached, landing entity",
                entity_id=entity_id,
                pos=(landing_x, landing_y, landing_h),
            )
            break  # Land entity at the last checked position

    # Calculate fall distance
    total_fall_distance = start_h - landing_h
    log.debug(
        "Fall calculated",
        entity_id=entity_id,
        distance=total_fall_distance,
        landing_pos=(landing_x, landing_y),
    )

    # Apply damage if fallen far enough
    if total_fall_distance > FALL_DAMAGE_THRESHOLD:
        damage = int(
            (total_fall_distance - FALL_DAMAGE_THRESHOLD) * FALL_DAMAGE_PER_UNIT_HEIGHT
        )
        if damage > 0:
            log.debug("Applying fall damage", entity_id=entity_id, damage=damage)
            current_hp = entity_reg.get_entity_component(entity_id, "hp")
            if current_hp is not None:
                new_hp = max(0, current_hp - damage)
                entity_reg.set_entity_component(entity_id, "hp", new_hp)
                # TODO: Check visibility for message?
                gs.add_message(f"You take {damage} falling damage!", (255, 0, 0))
                if new_hp <= 0:
                    # TODO: Handle entity death properly (drops, removal etc)
                    log.info("Entity died from fall damage", entity_id=entity_id)
                    name = (
                        entity_reg.get_entity_component(entity_id, "name")
                        or "Something"
                    )
                    handle_entity_death(entity_id, gs, f"The {name} dies from the fall!")
                    # Even if the entity dies, the fall action still counts as the turn's action
                    # No need to move the now-inactive entity
                    return True
            else:
                log.warning(
                    "Cannot apply fall damage, entity has no HP", entity_id=entity_id
                )

    # Move entity to landing spot only if it didn't die
    move_success = entity_reg.set_position(entity_id, Position(landing_x, landing_y))
    if not move_success:
        log.error(
            "Failed to set entity position after fall",
            entity_id=entity_id,
            landing_pos=(landing_x, landing_y),
        )
        # Entity might be stuck mid-air logically, but we still consumed the turn attempting to fall
        return True

    return True  # Falling consumes a turn


# --- Action Helper Functions ---
def _handle_player_move(dx: int, dy: int, gs: GameState, max_step: int) -> bool:
    """
    Attempts to move the player entity based on dx, dy.
    Handles bumping into entities (attack) and falling.
    Returns True if move, attack, or fall is successful and turn should be consumed.
    """
    log.debug("Attempting _handle_player_move", dx=dx, dy=dy)
    player_id = gs.player_id
    gm: GameMap = gs.game_map  # Add type hints for clarity
    entity_registry: EntityRegistry = gs.entity_registry

    current_pos = entity_registry.get_position(player_id)
    if current_pos is None:
        log.warning("Move failed: Player pos not found", player_id=player_id)
        return False
    current_x, current_y = current_pos
    new_x, new_y = current_x + dx, current_y + dy
    log_context = {
        "player_id": player_id,
        "from_pos": current_pos,
        "to_pos": (new_x, new_y),
    }

    # 1. Check Map Bounds
    if not gm.in_bounds(new_x, new_y):
        log.debug("Map bounds check FAILED")
        return False

    # 4. Check for Blocking Entities FIRST (before terrain checks, as bump/attack takes precedence)
    blocking_id = entity_registry.get_blocking_entity_at(new_x, new_y)
    if blocking_id is not None and blocking_id != player_id:
        log.debug("Entity blocking path, initiating attack", blocker_id=blocking_id)
        # --- Call Combat System ---
        try:
            # Call the imported combat_system's function
            # Assume attack always consumes turn for now
            combat_system.handle_melee_attack(player_id, blocking_id, gs)
            return True  # Attack happened, turn consumed
        except Exception as e_combat:
            log.error(
                "Exception during combat system call",
                error=str(e_combat),
                exc_info=True,
                attacker=player_id,
                defender=blocking_id,
            )
            gs.add_message("An error occurred during combat!", (255, 0, 0))
            return False  # Don't consume turn if combat system failed internally
        # --- End Call ---

    # If no blocking entity, proceed to terrain checks:

    # 2. Check Tile Walkability
    if not gm.is_walkable(new_x, new_y):
        log.debug("Walkability check FAILED")
        gs.add_message("That way is blocked by terrain.", (255, 127, 0))
        return False

    # 3. Check Height Difference & Fall
    try:
        # Use numpy's item() for potentially faster scalar access, default to array indexing
        h1 = int(gm.height_map.item(current_y, current_x))
        h2 = int(gm.height_map.item(new_y, new_x))
        delta_h = h2 - h1

        if abs(delta_h) > max_step:
            if delta_h > max_step:  # Step Up Too High
                log.debug(
                    "Height check FAILED (Step Up)", delta_h=delta_h, max_step=max_step
                )
                gs.add_message("That step is too high.", (255, 127, 0))
                return False
            else:  # delta_h < -max_step (Drop Too Steep) -> Handle Fall
                log.debug(
                    "Height check indicates fall", delta_h=delta_h, max_step=max_step
                )
                start_pos_h_tuple = (current_x, current_y, h1)
                fall_trigger_pos_tuple = (new_x, new_y, h2)
                # Call the fall handler - it returns True if turn is consumed
                return _handle_fall(
                    player_id, start_pos_h_tuple, fall_trigger_pos_tuple, gs, max_step
                )
        # If step is within limits (abs(delta_h) <= max_step), proceed normally
    except IndexError:
        log.error("IndexError during height check", **log_context)
        return False
    except Exception as e_h:
        log.error(
            "Exception during height check",
            error=str(e_h),
            exc_info=True,
            **log_context,
        )
        return False

    # 5. Perform Move (Only reached if no block, walkable, and height is okay)
    success = movement_system.try_move(player_id, dx, dy, gs)
    if success:
        log.debug("Player moved successfully", **log_context)
        return True  # Movement successful, turn consumed
    else:
        # This case should be rare if checks passed
        log.error("Player movement failed unexpectedly after checks", **log_context)
        return False


def _handle_player_pickup(gs: GameState) -> bool:
    """
    Attempts to pick up an item from the ground at the player's location.
    Returns True if pickup is successful and turn should be consumed.
    """
    player_id = gs.player_id
    player_pos = gs.player_position
    if player_pos is None:
        log.warning("Pickup action failed: player pos not found")
        return False  # Cannot act

    px, py = player_pos
    items_at_feet = gs.item_registry.get_items_at(px, py)

    if items_at_feet.height == 0:
        gs.add_message("There is nothing here to pick up.", (150, 150, 150))
        return False  # No items, no turn consumed

    # Simple: pick up the first item found (index 0 visually)
    # Consider adding logic for multiple items later (e.g., UI prompt)
    try:
        item_to_pickup = items_at_feet.row(0, named=True)
        item_id = item_to_pickup["item_id"]
        item_name = item_to_pickup["name"]
    except Exception as e:
        log.error(
            "Error accessing item data at feet",
            error=str(e),
            pos=(px, py),
            items_df_head=items_at_feet.head(1),
            exc_info=True,
        )
        gs.add_message("Error trying to pick up item.", (255, 0, 0))
        return False

    log.debug("Attempting pickup", item_id=item_id, name=item_name)

    # --- Inventory Capacity Check ---
    entity_reg = gs.entity_registry
    item_reg = gs.item_registry
    capacity = entity_reg.get_entity_component(player_id, "inventory_capacity")
    if capacity is not None:
        inventory_df = item_reg.get_entity_inventory(player_id)
        current_inventory_count = (
            int(inventory_df["quantity"].sum()) if inventory_df.height > 0 else 0
        )
        item_quantity = item_reg.get_item_component(item_id, "quantity") or 1
        if current_inventory_count + item_quantity > capacity:
            log.debug(
                "Inventory full, aborting pickup",
                player_id=player_id,
                capacity=capacity,
            )
            gs.add_message("Your inventory is full.", (255, 50, 50))
            return False  # Cannot pickup, no turn consumed

    # --- Weight Capacity Check ---
    item_quantity = item_reg.get_item_component(item_id, "quantity") or 1
    item_weight = (
        item_reg.get_item_static_attribute(item_id, "weight", default=0) * item_quantity
    )
    current_weight = (
        entity_reg.get_entity_component(player_id, "carried_weight") or 0
    )
    weight_capacity = (
        entity_reg.get_entity_component(player_id, "weight_capacity") or 0
    )
    if weight_capacity and current_weight + item_weight > weight_capacity:
        gs.add_message("You are carrying too much to pick that up.", (255, 50, 50))
        return False
    # --- End Capacity Check ---

    # Move item to inventory using ItemRegistry
    success = gs.item_registry.move_item(
        item_id=item_id,
        new_location="inventory",
        owner_entity_id=player_id,
        # x, y, equipped_slot, etc., are cleared automatically by move_item
    )
    if success:
        entity_reg.set_entity_component(
            player_id, "carried_weight", current_weight + item_weight
        )
        gs.add_message(f"You pick up the {item_name}.", (200, 200, 200))
        return True  # Pickup successful, turn consumed
    else:
        # Should be rare if item exists at location and capacity check passed
        gs.add_message(f"You can't pick up the {item_name}.", (255, 50, 50))
        log.error("Failed to move item to inventory during pickup", item_id=item_id)
        return False  # Failed pickup, no turn consumed


def _handle_player_drop(item_id_to_drop: int, gs: GameState) -> bool:
    """
    Attempts to drop an item from the player's inventory to the ground.
    Returns True if drop is successful and turn should be consumed.
    """
    player_id = gs.player_id
    player_pos = gs.player_position
    item_reg: ItemRegistry = gs.item_registry  # Add type hint
    entity_reg: EntityRegistry = gs.entity_registry

    if player_pos is None:
        log.warning("Drop action failed: player pos not found", player_id=player_id)
        gs.add_message("Cannot drop item: Your position is unknown.", (255, 0, 0))
        return False  # Cannot act

    px, py = player_pos

    # Verify the item is actually in the player's inventory
    item_owner = item_reg.get_item_component(item_id_to_drop, "owner_entity_id")
    item_loc = item_reg.get_item_component(item_id_to_drop, "location_type")

    if item_owner != player_id or item_loc != "inventory":
        log.warning(
            "Drop action failed: Item not in player inventory.",
            item_id=item_id_to_drop,
            owner=item_owner,
            location=item_loc,
            player_id=player_id,
        )
        gs.add_message("You aren't holding that item.", (255, 100, 100))
        return False  # Item not valid for dropping

    item_name = item_reg.get_item_component(item_id_to_drop, "name") or "the item"
    log.debug("Attempting drop", item_id=item_id_to_drop, name=item_name, pos=(px, py))

    # Use move_item to change location to ground at player's coords
    success = item_reg.move_item(
        item_id=item_id_to_drop,
        new_location="ground",
        x=px,
        y=py,
        # owner_entity_id, equipped_slot, etc., are cleared automatically by move_item
    )

    if success:
        item_quantity = item_reg.get_item_component(item_id_to_drop, "quantity") or 1
        item_weight = (
            item_reg.get_item_static_attribute(item_id_to_drop, "weight", default=0)
            * item_quantity
        )
        current_weight = (
            entity_reg.get_entity_component(player_id, "carried_weight") or 0
        )
        entity_reg.set_entity_component(
            player_id, "carried_weight", max(0, current_weight - item_weight)
        )
        gs.add_message(f"You drop the {item_name}.", (200, 200, 200))
        return True  # Drop successful, turn consumed
    else:
        # Should be rare if checks passed
        gs.add_message(f"You can't drop the {item_name}.", (255, 50, 50))
        log.error(
            "Failed to move item to ground during drop",
            item_id=item_id_to_drop,
            pos=(px, py),
        )
        return False  # Failed drop, no turn consumed


# --- Main Action Processing Function ---
def process_player_action(
    action: Dict[str, Any],
    gs: GameState,
    max_traversable_step: int,  # Pass necessary config directly
) -> bool:
    """
    Processes a player action dictionary, validates it, and performs the action.
    Calls appropriate handlers (movement, pickup, drop) or systems (equipment, effects).
    Returns True if the player successfully acted and consumed a turn.
    """
    action_type = action.get("type")
    player_id = gs.player_id  # Get player ID for convenience
    player_acted = False  # Default: action does not consume a turn unless successful
    log.debug(
        "ActionHandler: Processing action type",
        action_type=action_type,
        player_id=player_id,
        action_details=action,
    )

    match action_type:
        case "move":
            dx, dy = action.get("dx", 0), action.get("dy", 0)
            if dx != 0 or dy != 0:
                # _handle_player_move now includes attack and fall checks
                player_acted = _handle_player_move(dx, dy, gs, max_traversable_step)

        case "wait":
            gs.add_message("You wait.", (128, 128, 128))
            player_acted = True  # Waiting always consumes a turn
            log.debug("Player action", type="wait")

        case "pickup":
            player_acted = _handle_player_pickup(gs)

        case "drop":
            item_id = action.get("item_id")
            if item_id is None:
                log.warning("Drop action failed: No item_id specified.")
                gs.add_message("Drop what?", (255, 100, 100))
                player_acted = False
            else:
                # Ensure item_id is correctly typed if needed, though Polars handles it
                try:
                    item_id_int = int(item_id)  # Polars IDs are typically numeric
                    player_acted = _handle_player_drop(item_id_int, gs)
                except (ValueError, TypeError):
                    log.error(
                        "Drop action failed: Invalid item_id format.", item_id=item_id
                    )
                    gs.add_message("Invalid item specified.", (255, 0, 0))
                    player_acted = False

        case "use":
            item_id = action.get("item_id")
            target_info = action.get(
                "target", {}
            )  # Optional target pos/entity for effects
            if item_id is None:
                log.warning("Use action failed: No item_id specified")
                gs.add_message("Use what?", (255, 100, 100))
                player_acted = False
            else:
                # Assume the UI layer ensured the player *has* this item in inventory
                # and that the item *is* usable before generating this action.
                log.debug("Attempting to use item", item_id=item_id, target=target_info)

                # Prepare context for effect execution
                player_pos = gs.player_position
                if player_pos is None:
                    log.warning(
                        "Cannot execute 'use' action: Player position unknown.",
                        item_id=item_id,
                    )
                    gs.add_message(
                        "Cannot use item: Your position is unknown.", (255, 0, 0)
                    )
                    player_acted = False
                else:
                    context = {
                        "game_state": gs,
                        "rng": gs.rng_instance,
                        "source_entity_id": player_id,
                        "item_instance_id": item_id,  # Pass original (maybe non-int) ID if needed
                        "source_pos": player_pos,  # Use checked position
                        **target_info,  # Add target_pos, target_entity_id if present
                    }

                    # Get effects from template
                    template = gs.item_registry.get_template(
                        gs.item_registry.get_item_template_id(item_id)
                    )
                    if not template:  # Should ideally be caught by UI, but double check
                        log.error("Template missing for 'use' action", item_id=item_id)
                        player_acted = False
                    else:
                        effect_ids_to_run = template.get("effects", {}).get(
                            "active", []
                        ) + template.get("effects", {}).get("active_consumable", [])

                        if not effect_ids_to_run:
                            item_name = template.get("name", "item")
                            gs.add_message(
                                f"You can't use the {item_name}.", (255, 100, 100)
                            )
                            player_acted = False
                        else:
                            # Execute effects
                            executed_any = False
                            for effect_id in effect_ids_to_run:
                                # Shallow copy is usually fine.
                                if execute_effect(effect_id, context.copy()):
                                    executed_any = True
                                else:
                                    # Effect failed (cost, condition, target, error). Message should have been added.
                                    log.debug(
                                        "Effect execution failed during 'use' action",
                                        effect_id=effect_id,
                                        item_id=item_id,
                                    )
                                    # Optionally stop processing further effects if one fails?
                            # If at least one effect succeeded, action consumes a turn
                            player_acted = executed_any
                            if not executed_any:
                                log.debug(
                                    "No effects successfully executed for item use",
                                    item_id=item_id,
                                )
                                # Avoid adding "Nothing happens" message here, assume effect handlers/checks add specific failure messages.

        case "equip":
            item_id = action.get("item_id")  # Item ID from inventory
            if item_id is None:
                log.warning("Equip action failed: No item_id specified.")
                gs.add_message("Equip what?", (255, 100, 100))
                player_acted = False
            else:
                # Call the equipment system function
                player_acted = equipment_system.equip_item(player_id, item_id, gs)

        case "unequip":
            item_id = action.get("item_id")  # Item ID currently equipped
            if item_id is None:
                log.warning("Unequip action failed: No item_id specified.")
                gs.add_message("Unequip what?", (255, 100, 100))
                player_acted = False
            else:
                # Call the equipment system function
                player_acted = equipment_system.unequip_item(player_id, item_id, gs)

        case "attach":
            item_to_attach_id = action.get("item_to_attach_id")
            target_host_item_id = action.get("target_host_item_id")
            if item_to_attach_id is None or target_host_item_id is None:
                log.warning("Attach action failed: Missing item IDs.")
                gs.add_message("Attach what to what?", (255, 100, 100))
                player_acted = False
            else:
                # Call the equipment system function
                player_acted = equipment_system.attach_item(
                    player_id, item_to_attach_id, target_host_item_id, gs
                )

        case "detach":
            item_to_detach_id = action.get("item_to_detach_id")
            if item_to_detach_id is None:
                log.warning("Detach action failed: Missing item ID.")
                gs.add_message("Detach what?", (255, 100, 100))
                player_acted = False
            else:
                # Call the equipment system function
                player_acted = equipment_system.detach_item(
                    player_id, item_to_detach_id, gs
                )

        case _:
            log.warning(
                "Unknown action type received",
                received_action=action_type,
                action_details=action,
            )
            player_acted = False  # Unknown action doesn't consume a turn

    # --- Post-Action Processing ---
    if player_acted:
        log.debug("Player action resulted in turn", action_type=action_type)
        # Optionally advance game time, trigger NPC turns etc. handled elsewhere (e.g., MainLoop after this returns True)
    else:
        log.debug("Player action did not result in turn", action_type=action_type)

    return player_acted
