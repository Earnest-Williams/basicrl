# game/systems/combat_system.py
"""
Handles combat calculations and actions between entities.
"""
from typing import TYPE_CHECKING

import structlog

# Import roll_dice from its new location
from utils.helpers import roll_dice

if TYPE_CHECKING:
    from game_rng import GameRNG  # Assuming this is importable for type hint

    from game.entities.registry import EntityRegistry
    from game.game_state import GameState
    from game.items.registry import ItemRegistry

log = structlog.get_logger(__name__)

DEFAULT_UNARMED_DAMAGE = "1d2"  # Damage if attacker has no weapon


def handle_melee_attack(attacker_id: int, defender_id: int, gs: "GameState"):
    """
    Processes a melee attack from attacker_id to defender_id.
    Calculates damage, applies it, handles messages, and checks for death.
    Currently assumes the action is valid and consumes a turn.
    """
    entity_reg: "EntityRegistry" = gs.entity_registry
    item_reg: "ItemRegistry" = gs.item_registry
    rng: "GameRNG" = gs.rng_instance  # Get RNG from GameState

    attacker_name = entity_reg.get_entity_component(attacker_id, "name") or "Attacker"
    defender_name = entity_reg.get_entity_component(defender_id, "name") or "Defender"
    log.debug(
        "Handling melee attack",
        attacker=attacker_name,
        defender=defender_name,
        attacker_id=attacker_id,
        defender_id=defender_id,
    )

    # --- Determine Attacker's Damage ---
    damage_dice = DEFAULT_UNARMED_DAMAGE
    weapon_name = "unarmed"

    # Find equipped weapon
    equipped_ids = entity_reg.get_equipped_ids(attacker_id)
    if equipped_ids:
        # Check main_hand first, then off_hand (simple priority for now)
        # TODO: Refine to handle two-handed weapons or dual wielding later
        main_hand_weapon_id = None
        # Query items registry for equipped items owned by attacker
        equipped_items = item_reg.get_entity_equipped(attacker_id).filter(
            pl.col("item_id").is_in(equipped_ids)
        )
        if equipped_items.height > 0:
            main_hand_item = equipped_items.filter(
                pl.col("equipped_slot") == "main_hand"
            ).row(0, named=True, default=None)
            if main_hand_item:
                main_hand_weapon_id = main_hand_item.get("item_id")

            # Simple fallback to any equipped weapon if main hand empty (could be off_hand weapon)
            if not main_hand_weapon_id:
                any_weapon = equipped_items.filter(pl.col("item_type") == "Weapon").row(
                    0, named=True, default=None
                )  # Assuming 'item_type' exists
                if any_weapon:
                    main_hand_weapon_id = any_weapon.get(
                        "item_id"
                    )  # Treat as primary weapon for now

        if main_hand_weapon_id:
            # Retrieve damage_dice attribute from the weapon's template
            weapon_damage_dice_attr = item_reg.get_item_static_attribute(
                main_hand_weapon_id, "damage_dice", default=None
            )
            if weapon_damage_dice_attr:
                damage_dice = weapon_damage_dice_attr
                weapon_name = (
                    item_reg.get_item_component(main_hand_weapon_id, "name") or "weapon"
                )
                log.debug("Attacker using weapon", weapon=weapon_name, dice=damage_dice)
            else:
                log.warning(
                    "Equipped weapon has no damage_dice attribute",
                    item_id=main_hand_weapon_id,
                )
    else:
        log.debug("Attacker is unarmed")

    # --- Calculate Damage ---
    # TODO: Factor in attacker stats (e.g., strength)
    # TODO: Factor in defender stats (e.g., defense, armor class)
    raw_damage = roll_dice(damage_dice, rng)
    final_damage = max(0, raw_damage)  # Ensure damage isn't negative

    # --- Apply Damage & Check Death ---
    defender_hp = entity_reg.get_entity_component(defender_id, "hp")
    if defender_hp is None:
        log.error("Defender missing HP component", defender_id=defender_id)
        return  # Cannot apply damage

    new_hp = max(0, defender_hp - final_damage)
    damage_dealt = defender_hp - new_hp

    log.debug(
        "Damage calculated",
        raw=raw_damage,
        final=final_damage,
        dealt=damage_dealt,
        defender_hp_old=defender_hp,
        defender_hp_new=new_hp,
    )

    # Add Combat Messages
    # TODO: Visibility checks
    attack_msg = ""
    damage_msg = ""
    hit_color = (200, 200, 200)  # Default color
    damage_color = (255, 255, 0)  # Yellow for damage

    if attacker_id == gs.player_id:
        attack_msg = f"You attack the {defender_name} with your {weapon_name}!"
        if damage_dealt > 0:
            damage_msg = f"You hit for {damage_dealt} damage!"
            damage_color = (0, 255, 0)  # Green for player dealing damage
        else:
            damage_msg = "You miss!"
            damage_color = (150, 150, 150)
    elif defender_id == gs.player_id:
        attack_msg = f"The {attacker_name} attacks you with its {weapon_name}!"
        if damage_dealt > 0:
            damage_msg = f"You are hit for {damage_dealt} damage!"
            damage_color = (255, 0, 0)  # Red for player taking damage
        else:
            damage_msg = f"The {attacker_name} misses!"
            damage_color = (150, 150, 150)
    else:  # Mob vs Mob
        attack_msg = f"The {attacker_name} attacks the {defender_name}!"
        if damage_dealt > 0:
            damage_msg = f"It hits for {damage_dealt} damage."
        else:
            damage_msg = "It misses."

    if attack_msg:
        gs.add_message(attack_msg, hit_color)
    if damage_msg:
        gs.add_message(damage_msg, damage_color)

    # Update Defender HP
    if damage_dealt > 0:
        update_success = entity_reg.set_entity_component(defender_id, "hp", new_hp)
        if not update_success:
            log.error("Failed to set defender HP after attack", defender_id=defender_id)
            # Continue to death check anyway, HP might conceptually be 0

    # Handle Death [Source [source 53]]
    if new_hp <= 0:
        log.info(f"{defender_name} died.", defender_id=defender_id)
        gs.add_message(f"The {defender_name} dies!", (255, 100, 100))  # Death color
        # TODO: Handle XP, item drops, remove body etc.
        entity_reg.delete_entity(defender_id)  # Mark inactive
