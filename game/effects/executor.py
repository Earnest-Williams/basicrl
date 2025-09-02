# game/effects/executor.py
from typing import TYPE_CHECKING, Any, Dict

import structlog

# Import handlers dictionary
from .handlers import EFFECT_LOGIC_HANDLERS  # Keep this import

if TYPE_CHECKING:
    # No longer need ItemRegistry specific import here, accessed via GameState
    # from ..game.items.registry import ItemRegistry
    # Add EntityRegistry import for type hinting if needed, though accessed via gs
    from ..game.entities.registry import EntityRegistry
    from ..game_state import GameState

log = structlog.get_logger()

# --- Helper function stubs ---


def _check_targeting(effect_definition: dict, context: dict) -> bool:
    # ... (targeting logic remains the same) ...
    log.debug("Targeting check skipped (placeholder)")
    return True


def _check_and_deduct_costs(effect_definition: dict, context: Dict[str, Any]) -> bool:
    """
    Checks if costs (item charges, mana, fullness, etc.) can be met
    and deducts them if possible.
    """
    gs: "GameState" | None = context.get("game_state")  # Type hint with | None
    source_id = context.get("source_entity_id")
    # Still needed for item_charge cost
    item_id = context.get("item_instance_id")

    costs = effect_definition.get("cost", [])
    if not costs:
        return True  # No cost defined

    if gs is None:
        log.error("Cost check failed: game_state missing from context")
        return False

    # Ensure entity_registry is available
    entity_registry: "EntityRegistry" = gs.entity_registry
    if entity_registry is None:
        log.error("Cost check failed: entity_registry missing from game_state")
        return False

    # --- Check all costs first ---
    can_pay = True
    failure_reason = ""
    for cost_info in costs:
        cost_type = cost_info.get("type")
        amount = cost_info.get("amount", 1)  # Default amount to 1 if missing
        if amount <= 0:
            continue  # Skip zero or negative costs

        log.debug("Checking cost", type=cost_type, amount=amount)

        if cost_type == "item_charge":
            if item_id is None:
                log.warning(
                    "Cost check failed: item_charge cost requires item_instance_id"
                )
                can_pay = False
                failure_reason = "Item charge required but no item specified"
                break
            # Ensure item_registry exists
            if gs.item_registry is None:
                log.error(
                    "Cost check failed: item_registry missing for item_charge cost"
                )
                can_pay = False
                failure_reason = "Internal error: ItemRegistry missing"
                break
            current_charge = gs.item_registry.get_item_component(
                item_id, "current_charge"
            )
            if current_charge is None or current_charge < amount:
                log.debug(
                    "Cost check failed: Insufficient item charge",
                    item=item_id,
                    needed=amount,
                    has=current_charge,
                )
                can_pay = False
                failure_reason = "Insufficient item charge"
                break

        elif cost_type in ("mana", "fullness"):  # Handle generic resources
            if source_id is None:
                log.warning(
                    f"Cost check failed: {
                        cost_type} cost requires source_entity_id"
                )
                can_pay = False
                failure_reason = (
                    f"{cost_type.capitalize()} cost requires a source entity"
                )
                break
            current_value = entity_registry.get_entity_component(source_id, cost_type)
            if current_value is None or current_value < amount:
                log.debug(
                    f"Cost check failed: Insufficient {cost_type}",
                    source=source_id,
                    needed=amount,
                    has=current_value,
                )
                can_pay = False
                failure_reason = f"Insufficient {cost_type}"
                break
        # --- Add elif for other cost types (e.g., hp) here ---
        # elif cost_type == "hp": ...
        else:
            log.warning("Cost check failed: Unknown cost type", type=cost_type)
            can_pay = False
            failure_reason = f"Unknown cost type '{cost_type}'"
            break

    if not can_pay:
        # Add message to player if the check failed for the player
        if source_id == gs.player_id:
            # Provide a more specific message if possible
            message = failure_reason or "You lack the required resources."
            gs.add_message(message.capitalize() + ".", (255, 100, 100))
        return False

    # --- Deduct costs (only if all checks passed) ---
    log.debug("Deducting costs", costs=costs)
    deduction_failed = False
    for cost_info in costs:
        cost_type = cost_info.get("type")
        amount = cost_info.get("amount", 1)
        if amount <= 0:
            continue

        if cost_type == "item_charge":
            # Re-get value just before setting for safety
            current_charge = gs.item_registry.get_item_component(
                item_id, "current_charge"
            )
            # Check again in case of race condition (if threading added later)
            if current_charge is None or current_charge < amount:
                log.error(
                    "Charge deduction failed: charge changed between check and deduct",
                    item=item_id,
                )
                deduction_failed = True
                break  # Abort deduction
            success = gs.item_registry.set_item_component(
                item_id, "current_charge", current_charge - amount
            )
            if not success:
                log.error("Failed to deduct item charge", item=item_id)
                deduction_failed = True
                break

        elif cost_type in ("mana", "fullness"):  # Handle generic resources
            # Re-get value
            current_value = entity_registry.get_entity_component(source_id, cost_type)
            if current_value is None or current_value < amount:
                log.error(
                    f"{cost_type.capitalize(
                    )} deduction failed: value changed between check and deduct",
                    source=source_id,
                )
                deduction_failed = True
                break
            new_value = current_value - amount
            success = entity_registry.set_entity_component(
                source_id, cost_type, new_value
            )
            if not success:
                log.error(f"Failed to deduct {cost_type}", source=source_id)
                deduction_failed = True
                break
            log.debug(
                f"Deducted {cost_type}",
                source=source_id,
                amount=amount,
                new_value=new_value,
            )
        # --- Add elif for other cost deductions here ---

    # --- IMPORTANT: Handle Deduction Failure ---
    # If any deduction failed, we have a problem. The state might be inconsistent.
    # Ideally, this would involve a transaction system to roll back previous deductions.
    # For now, we log a critical error and return False, potentially leaving the
    # game state inconsistent (e.g., charge deducted but mana deduction failed).
    if deduction_failed:
        log.critical(
            "Deduction phase failed mid-way! Game state may be inconsistent.",
            failed_cost_type=cost_type,
            source=source_id,
            item=item_id,
        )
        # Cannot easily undo previous deductions without more complex state management.
        return False  # Indicate overall failure

    log.debug("Costs deducted successfully")
    return True


def _check_conditions(effect_definition: dict, context: Dict[str, Any]) -> bool:
    # ... (conditions logic remains the same) ...
    log.debug("All conditions met")  # Assuming it passes for now
    return True


def _is_consumable_effect(effect_id: str, context: Dict[str, Any]) -> bool:
    # ... (consumption check logic remains the same) ...
    item_id = context.get("item_instance_id")
    if not item_id:
        return False
    gs: "GameState" | None = context.get("game_state")
    if gs is None or gs.item_registry is None:
        return False
    template_id = gs.item_registry.get_item_template_id(item_id)
    if not template_id:
        return False
    template = gs.item_registry.get_template(template_id)
    if not template:
        return False
    if effect_id in template.get("effects", {}).get("active_consumable", []):
        return True
    return False


def _consume_item(item_id: int, context: Dict[str, Any]):
    # ... (item consumption logic remains the same) ...
    gs: "GameState" | None = context.get("game_state")
    if gs is None or gs.item_registry is None:
        return
    quantity = gs.item_registry.get_item_component(item_id, "quantity")
    if quantity is None:
        log.warning("Cannot consume item: quantity is None", item_id=item_id)
        return
    if quantity > 1:
        log.debug("Decrementing consumable quantity", item_id=item_id, old_qty=quantity)
        gs.item_registry.set_item_component(item_id, "quantity", quantity - 1)
    else:
        log.debug("Deleting consumed item", item_id=item_id)
        gs.item_registry.delete_item(item_id)


# --- Main Executor ---


def execute_effect(effect_id: str, context: Dict[str, Any]) -> bool:
    """
    Executes a defined effect based on its ID and the provided context.
    Includes checks for targeting, costs, and conditions.
    """
    log.debug(
        "Executing effect", effect_id=effect_id, context_keys=list(context.keys())
    )

    gs: "GameState" | None = context.get("game_state")  # Use | None
    if gs is None:
        log.error(
            "execute_effect failed: 'game_state' missing from context",
            effect_id=effect_id,
        )
        return False

    effect_definition = gs.effect_definitions.get(effect_id)
    if not effect_definition:
        log.warning("execute_effect failed: Unknown effect_id", effect_id=effect_id)
        return False

    # --- Execute Pre-Checks ---
    if not _check_targeting(effect_definition, context):
        log.debug("Effect targeting check failed", effect_id=effect_id)
        return False

    if not _check_conditions(effect_definition, context):
        log.debug("Effect condition check failed", effect_id=effect_id)
        return False

    # --- Check and Deduct Costs ---
    # This function now potentially adds messages on failure
    if not _check_and_deduct_costs(effect_definition, context):
        log.debug("Effect cost check failed or deduction failed", effect_id=effect_id)
        # Message is now added inside _check_and_deduct_costs if needed
        return False

    # --- Find and Execute Logic Handler ---
    logic_handler_id = effect_definition.get("logic_handler")
    if not logic_handler_id:
        log.warning("Effect definition missing 'logic_handler'", effect_id=effect_id)
        return False

    handler_func = EFFECT_LOGIC_HANDLERS.get(logic_handler_id)
    if not handler_func:
        log.error(
            "execute_effect failed: Unknown logic_handler in registry",
            effect_id=effect_id,
            logic_handler=logic_handler_id,
        )
        return False

    # --- Execute Core Logic ---
    try:
        params = effect_definition.get("params", {})
        handler_func(context, params)
        log.debug(
            "Effect handler executed successfully",
            effect_id=effect_id,
            handler=logic_handler_id,
        )

        # --- Handle Consumption (After successful execution) ---
        if _is_consumable_effect(effect_id, context):
            item_id = context.get("item_instance_id")
            if item_id:
                _consume_item(item_id, context)

        return True  # Indicate successful execution attempt

    except Exception as e:
        log.error(
            "Exception during effect handler execution",
            effect_id=effect_id,
            handler=logic_handler_id,
            error=str(e),
            exc_info=True,
        )
        # --- Rollback Consideration ---
        # Since deductions happened before execution, a failure here leaves state inconsistent.
        # A true rollback mechanism would be needed for atomicity.
        # For now, log the error. The effect failed, but costs were paid.
        gs.add_message(
            "An unexpected error occurred!", (255, 0, 0)
        )  # Generic error message
        return False  # Indicate failure
