# game/effects/handlers.py
# Contains the actual Python functions that implement effect logic.

import re
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import structlog
import polars as pl

# Imports for type hinting and GameRNG
from game_rng import GameRNG
from ..world import line_of_sight
from ..systems.death_system import handle_entity_death
from magic.models import Art, Substance

# Import sound system for audio feedback
try:
    from ..systems.sound import handle_event, play_sound
except ImportError:
    # Fallback if sound system is not available
    def handle_event(event_name: str, context=None):
        pass

    def play_sound(effect_name: str, context=None):
        return False


if TYPE_CHECKING:
    # Corrected relative import path assuming handlers.py is inside effects folder
    from ..game_state import GameState

log = structlog.get_logger()


def is_visible_to_player(entity_id: int | None, gs: "GameState") -> bool:
    """Return True if ``entity_id`` is within the player's current FOV."""
    if entity_id is None:
        return False
    pos = gs.entity_registry.get_position(entity_id)
    if not pos:
        return False
    if not gs.game_map.in_bounds(pos.x, pos.y):
        return False
    return bool(gs.game_map.visible[pos.y, pos.x])


# --- Dice Rolling Utility ---
DICE_PATTERN = re.compile(r"(\d+)?d(\d+)(?:([+-])(\d+))?")


def _roll_dice(dice_str: str | None, rng: GameRNG | None) -> int:
    if not dice_str:
        return 0
    if rng is None:
        log.warning("Dice roll attempted without RNG context!")
        return 0
    match = DICE_PATTERN.match(dice_str)
    if match:
        num_dice_str, sides_str, operator, bonus_str = match.groups()
        num_dice = int(num_dice_str) if num_dice_str else 1
        sides = int(sides_str)
        bonus = int(f"{operator}{bonus_str}") if operator and bonus_str else 0
        if sides <= 0:
            return bonus
        if num_dice <= 0:
            return bonus
        roll_total = sum(rng.get_int(1, sides) for _ in range(num_dice))
        return roll_total + bonus
    else:
        try:
            return int(dice_str)
        except ValueError:
            log.error("Invalid dice string format", dice_str=dice_str)
            return 0


# --- AOE Helper ---
def _get_entities_in_aoe(
    center_pos: Tuple[int, int], radius: int, gs: "GameState"
) -> List[int]:
    """Finds active entity IDs within a specified radius of a center point."""
    target_entities: List[int] = []
    radius_sq = radius * radius
    cx, cy = center_pos

    active_entities_df = gs.entity_registry.get_active_entities()
    if active_entities_df.height == 0:
        return []

    # Filter entities within the bounding box first (minor optimization)
    potential_targets = active_entities_df.filter(
        (pl.col("x") >= cx - radius)
        & (pl.col("x") <= cx + radius)
        & (pl.col("y") >= cy - radius)
        & (pl.col("y") <= cy + radius)
    )

    # Precise distance check
    if potential_targets.height > 0:
        potential_targets = potential_targets.with_columns(
            dist_sq=((pl.col("x") - cx) ** 2 + (pl.col("y") - cy) ** 2)
        ).filter(pl.col("dist_sq") <= radius_sq)
        candidate_ids = potential_targets["entity_id"].to_list()

        transparency = gs.game_map.transparent
        for entity_id in candidate_ids:
            pos = gs.entity_registry.get_position(entity_id)
            if not pos:
                continue
            if line_of_sight(cy, cx, pos.y, pos.x, transparency):
                target_entities.append(entity_id)

    log.debug(
        "AOE Query results",
        center=center_pos,
        radius=radius,
        count=len(target_entities),
        targets=target_entities,
    )
    return target_entities


# --- Handler Functions ---


def heal_target(context: Dict[str, Any], params: Dict[str, Any]):
    """Heals the target entity."""
    gs: "GameState" = context.get("game_state")
    rng: GameRNG | None = context.get("rng")
    target_id = context.get("target_entity_id", context.get("source_entity_id"))
    if gs is None or target_id is None or rng is None:
        log.warning(
            "Heal effect missing context", target_id=target_id, has_rng=bool(rng)
        )
        return

    base_heal = params.get("base_heal", 0)
    dice_str = params.get("dice")
    heal_amount = base_heal + _roll_dice(dice_str, rng)
    if heal_amount <= 0:
        return

    current_hp = gs.entity_registry.get_entity_component(target_id, "hp")
    max_hp = gs.entity_registry.get_entity_component(target_id, "max_hp")
    if current_hp is None or max_hp is None:
        log.warning("Heal target missing HP components", target_id=target_id)
        return

    new_hp = min(max_hp, current_hp + heal_amount)
    amount_healed = new_hp - current_hp
    if amount_healed > 0:
        success = gs.entity_registry.set_entity_component(target_id, "hp", new_hp)
        if success:
            log.debug(
                "Healed entity",
                target_id=target_id,
                amount=amount_healed,
                new_hp=new_hp,
            )
            target_name = (
                gs.entity_registry.get_entity_component(target_id, "name")
                or "Something"
            )
            target_pos = gs.entity_registry.get_position(target_id)
            player_can_see = target_id == gs.player_id or (
                target_pos and gs.game_map.visible[target_pos.y, target_pos.x]
            )

            # Play healing sound effect
            sound_context = {
                "target": "player" if target_id == gs.player_id else "other",
                "amount": amount_healed,
                "visible": player_can_see,
            }
            handle_event("heal_target", sound_context)

            if player_can_see:
                if target_id == gs.player_id:
                    gs.add_message(f"You heal for {amount_healed} HP.", (0, 255, 0))
                else:
                    gs.add_message(
                        f"The {target_name} heals for {amount_healed} HP.",
                        (0, 180, 0),
                    )
        else:
            log.error("Failed to set new HP for heal target", target_id=target_id)


def modify_resource(context: Dict[str, Any], params: Dict[str, Any]):
    """Modifies a generic entity resource (e.g., fullness, mana). Requires component exists."""
    gs: "GameState" = context.get("game_state")
    target_id = context.get("target_entity_id", context.get("source_entity_id"))
    resource_name = params.get("resource")
    change_dice = params.get("dice")  # Allow dice roll for change
    base_change = params.get("base_change", 0)
    rng: GameRNG | None = context.get("rng")

    if gs is None or target_id is None or not resource_name:
        log.warning(
            "Modify resource missing required info", target=target_id, res=resource_name
        )
        return

    change = base_change + _roll_dice(change_dice, rng)
    if change == 0:
        return  # No change

    # --- !!! IMPORTANT: Assumes components like 'mana', 'max_mana', 'fullness' exist !!! ---
    # --- !!! Need to add these to ENTITY_SCHEMA in registry.py                  !!! ---
    current_value = gs.entity_registry.get_entity_component(target_id, resource_name)
    max_value_component = f"max_{resource_name}"  # Convention: e.g., max_mana
    max_value = gs.entity_registry.get_entity_component(target_id, max_value_component)

    if current_value is None:
        log.warning(
            f"Target entity missing resource component '{resource_name}'",
            target_id=target_id,
        )
        return

    try:
        # Ensure types allow addition (e.g., handle None for max_value)
        numeric_current = float(current_value)  # Cast to float for flexibility
        new_value = numeric_current + change

        if max_value is not None:
            numeric_max = float(max_value)
            new_value = min(numeric_max, new_value)

        new_value = max(0.0, new_value)  # Assume resources don't go below 0

        success = gs.entity_registry.set_entity_component(
            target_id, resource_name, new_value
        )
        if success:
            log.debug(
                "Modified resource",
                target=target_id,
                resource=resource_name,
                change=change,
                new_value=new_value,
            )
        else:
            log.error(
                "Failed to set new resource value",
                target=target_id,
                resource=resource_name,
            )
    except (TypeError, ValueError) as e:
        log.error(
            "Type/Value error during resource modification",
            target=target_id,
            resource=resource_name,
            current_val=current_value,
            change=change,
            error=str(e),
        )
    except Exception as e:
        log.error(
            "Error modifying resource",
            target=target_id,
            resource=resource_name,
            error=str(e),
            exc_info=True,
        )


def recall_ammo(context: Dict[str, Any], params: Dict[str, Any]):
    """Returns a projectile item to its owner's inventory."""
    gs: "GameState" = context.get("game_state")
    projectile_item_id = context.get("projectile_item_id")
    source_entity_id = context.get("source_entity_id")
    if gs is None or projectile_item_id is None or source_entity_id is None:
        log.warning(
            "Recall ammo missing required context",
            item=projectile_item_id,
            owner=source_entity_id,
        )
        return

    item_active = gs.item_registry.get_item_component(projectile_item_id, "is_active")
    item_loc = gs.item_registry.get_item_component(projectile_item_id, "location_type")

    if not item_active or item_loc != "ground":
        log.debug(
            "Cannot recall ammo: not active or not on ground",
            item=projectile_item_id,
            loc=item_loc,
            active=item_active,
        )
        return

    success = gs.item_registry.move_item(
        item_id=projectile_item_id,
        new_location="inventory",
        owner_entity_id=source_entity_id,
    )
    if success:
        log.info(
            "Ammo recalled successfully",
            item=projectile_item_id,
            owner=source_entity_id,
        )

        # Play recall sound effect
        sound_context = {
            "item_type": "projectile",
            "target": "player" if source_entity_id == gs.player_id else "other",
        }
        handle_event("recall_ammo", sound_context)

        if source_entity_id == gs.player_id:
            item_name = (
                gs.item_registry.get_item_component(projectile_item_id, "name")
                or "projectile"
            )
            gs.add_message(f"Your {item_name} mysteriously returns!", (150, 150, 255))
    else:
        log.error(
            "Failed to move recalled ammo to inventory",
            item=projectile_item_id,
            owner=source_entity_id,
        )


def apply_status(context: Dict[str, Any], params: Dict[str, Any]):
    """Applies a status effect to the target entity."""
    gs: "GameState" = context.get("game_state")
    target_id = context.get("target_entity_id", context.get("source_entity_id"))
    status_id = params.get("status")
    duration_dice = params.get("duration_dice")
    base_duration = params.get("base_duration", 1)
    intensity_dice = params.get("intensity_dice")
    base_intensity = params.get("base_intensity")  # Can be None
    rng: GameRNG | None = context.get("rng")

    if gs is None or target_id is None or not status_id:
        log.warning(
            "ApplyStatus missing required info", target=target_id, status=status_id
        )
        return

    duration = base_duration + _roll_dice(duration_dice, rng)
    intensity = None
    if base_intensity is not None or intensity_dice:
        intensity_roll = _roll_dice(intensity_dice, rng)
        intensity = (base_intensity or 0) + intensity_roll
        # Ensure intensity is float if calculated
        intensity = float(intensity) if intensity is not None else None

    log.debug(
        "Applying status",
        target=target_id,
        status=status_id,
        dur=duration,
        intens=intensity,
    )

    current_statuses_list = gs.entity_registry.get_entity_component(
        target_id, "status_effects"
    )
    if current_statuses_list is None:
        log.error(
            "Target entity missing 'status_effects' component", target_id=target_id
        )
        return

    updated_statuses: List[Dict[str, Any]] = []
    found_existing = False

    for status_dict in current_statuses_list:
        if not isinstance(status_dict, dict):
            log.warning(
                "Skipping invalid status entry", entry=status_dict, target_id=target_id
            )
            continue

        if status_dict.get("id") == status_id:
            found_existing = True
            # Update logic: Max duration, replace intensity if new one provided & not None
            new_duration = max(status_dict.get("duration", 0), duration)
            new_intensity = (
                intensity if intensity is not None else status_dict.get("intensity")
            )

            updated_status = {"id": status_id, "duration": new_duration}
            if new_intensity is not None:
                updated_status["intensity"] = float(new_intensity)  # Ensure float

            updated_statuses.append(updated_status)
            log.debug(
                "Updated existing status",
                status_id=status_id,
                new_dur=new_duration,
                new_intens=new_intensity,
            )
        else:
            if "intensity" in status_dict and status_dict["intensity"] is not None:
                status_dict["intensity"] = float(status_dict["intensity"])
            updated_statuses.append(status_dict)

    if not found_existing:
        new_status = {"id": status_id, "duration": duration}
        if intensity is not None:
            new_status["intensity"] = float(intensity)
        updated_statuses.append(new_status)
        log.debug(
            "Added new status", status_id=status_id, dur=duration, intens=intensity
        )

    success = gs.entity_registry.set_entity_component(
        target_id, "status_effects", updated_statuses
    )
    if not success:
        log.error(
            "Failed to apply status effect component update",
            target_id=target_id,
            status_id=status_id,
        )
    else:
        # Messaging based on visibility
        target_name = (
            gs.entity_registry.get_entity_component(target_id, "name") or "Something"
        )
        if target_id == gs.player_id:
            gs.add_message(f"You feel {status_id}!", (255, 50, 50))
        else:
            msg = f"The {target_name} is afflicted by {status_id}!"
            if is_visible_to_player(target_id, gs):
                gs.add_message(msg, (200, 100, 100))
            else:
                gs.queue_message(target_id, msg, (200, 100, 100))


# --- Implemented AOE ---
def apply_status_in_aoe(context: Dict[str, Any], params: Dict[str, Any]):
    """Applies a status effect to all valid targets within an AOE."""
    gs: "GameState" = context.get("game_state")
    center_pos = context.get("target_pos")
    radius = params.get("radius", 1)
    if gs is None or center_pos is None or radius <= 0:
        log.warning(
            "ApplyStatusAOE missing context/params", center=center_pos, radius=radius
        )
        return

    log.debug("Applying status in AOE", center=center_pos, radius=radius, params=params)
    targets = _get_entities_in_aoe(center_pos, radius, gs)

    # Create a context copy for each target to avoid side-effects
    base_context = context.copy()
    for target_id in targets:
        target_context = base_context.copy()
        target_context["target_entity_id"] = target_id
        # Call the single-target apply_status handler
        apply_status(target_context, params)


# --- Implemented Damage ---
def deal_damage(context: Dict[str, Any], params: Dict[str, Any]):
    """Deals damage to a target entity."""
    gs: "GameState" = context.get("game_state")
    rng: GameRNG | None = context.get("rng")
    source_id = context.get("source_entity_id")  # Optional: for messages/XP
    target_id = context.get("target_entity_id")
    # e.g., physical, fire, magic
    damage_type = params.get("damage_type", "physical")
    dice_str = params.get("dice")
    base_damage = params.get("base_damage", 0)

    if gs is None or target_id is None or rng is None:
        log.warning("DealDamage missing context", target=target_id, has_rng=bool(rng))
        return

    raw_damage = base_damage + _roll_dice(dice_str, rng)
    if raw_damage <= 0:
        log.debug("Damage calculated as zero or less", raw=raw_damage)
        return

    resistances: Dict[str, float] = {}
    vulnerabilities: Dict[str, float] = {}
    try:
        resistances = (
            gs.entity_registry.get_entity_component(target_id, "resistances") or {}
        )
    except ValueError:
        pass
    try:
        vulnerabilities = (
            gs.entity_registry.get_entity_component(target_id, "vulnerabilities") or {}
        )
    except ValueError:
        pass

    multiplier = 1.0
    if isinstance(resistances, dict):
        multiplier *= 1 - float(resistances.get(damage_type, 0))
    if isinstance(vulnerabilities, dict):
        multiplier *= 1 + float(vulnerabilities.get(damage_type, 0))
    final_damage = max(0, int(raw_damage * multiplier))

    current_hp = gs.entity_registry.get_entity_component(target_id, "hp")
    if current_hp is None:
        log.warning("DealDamage target missing HP component", target_id=target_id)
        return

    new_hp = max(0, current_hp - final_damage)  # Ensure HP doesn't go below 0
    amount_damaged = current_hp - new_hp

    if amount_damaged > 0:
        log.debug(
            "Applying damage",
            target=target_id,
            damage=amount_damaged,
            new_hp=new_hp,
            type=damage_type,
        )
        success = gs.entity_registry.set_entity_component(target_id, "hp", new_hp)
        if success:
            source_name = "Something"
            if source_id:
                source_name = (
                    gs.entity_registry.get_entity_component(source_id, "name")
                    or "Something"
                )
            target_name = (
                gs.entity_registry.get_entity_component(target_id, "name")
                or "Something"
            )

            # Visibility using helper
            source_visible = is_visible_to_player(source_id, gs)
            target_visible = is_visible_to_player(target_id, gs)
            player_can_see = (
                (source_id == gs.player_id)
                or (target_id == gs.player_id)
                or source_visible
                or target_visible
            )

            # Play damage sound effect
            sound_context = {
                "target": "player" if target_id == gs.player_id else "other",
                "damage_type": damage_type,
                "amount": amount_damaged,
                "visible": player_can_see,
                "fatal": new_hp <= 0,
            }
            handle_event("deal_damage", sound_context)

            # Messaging
            if target_id == gs.player_id:
                gs.add_message(
                    f"The {source_name} hits you for {amount_damaged} damage!",
                    (255, 0, 0),
                )
            elif source_id == gs.player_id:
                gs.add_message(
                    f"You hit the {target_name} for {amount_damaged} damage!",
                    (0, 255, 0),
                )
            elif player_can_see:
                gs.add_message(
                    f"The {source_name} hits the {target_name} for {amount_damaged}.",
                    (200, 200, 0),
                )
            else:
                gs.queue_message(
                    target_id,
                    f"The {source_name} hits the {target_name} for {amount_damaged}.",
                    (200, 200, 0),
                )

            # --- Handle Death ---
            if new_hp <= 0:
                log.info(
                    "Entity died", target_id=target_id, name=target_name, hp=new_hp
                )
                handle_entity_death(target_id, gs)
                # Maybe add 'target_is_living': False to context for trigger checks?
                context["target_is_living"] = False
        else:
            log.error("Failed to set new HP for damage target", target_id=target_id)


# --- Implemented AOE Damage ---
def deal_damage_in_aoe(context: Dict[str, Any], params: Dict[str, Any]):
    """Deals damage to all valid targets within an AOE."""
    gs: "GameState" = context.get("game_state")
    center_pos = context.get("target_pos")
    radius = params.get("radius", 1)
    # Pass source for damage messages
    source_id = context.get("source_entity_id")

    if gs is None or center_pos is None or radius <= 0:
        log.warning(
            "DealDamageAOE missing context/params", center=center_pos, radius=radius
        )
        return

    log.debug("Dealing damage in AOE", center=center_pos, radius=radius, params=params)
    targets = _get_entities_in_aoe(center_pos, radius, gs)

    # Create a context copy for each target
    base_context = context.copy()
    for target_id in targets:
        # Don't hit self unless explicitly allowed? (Assume no self-hit for now)
        if source_id is not None and target_id == source_id:
            continue

        target_context = base_context.copy()
        target_context["target_entity_id"] = target_id
        # Call the single-target deal_damage handler
        deal_damage(target_context, params)


# --- Implemented Dig Tunnel (minor refinement) ---
def dig_tunnel(context: Dict[str, Any], params: Dict[str, Any]):
    gs: "GameState" = context.get("game_state")
    source_entity_id = context.get("source_entity_id")
    direction = context.get("target_direction")
    if gs is None or source_entity_id is None or direction is None:
        log.warning("DigTunnel missing context")
        return

    # Corrected import path
    try:
        from ..world.game_map import TILE_ID_FLOOR, TILE_ID_WALL
    except ImportError:
        log.error("Could not import TILE_ID constants for digging.")
        return

    start_pos = gs.entity_registry.get_position(source_entity_id)
    if not start_pos:
        log.warning("Dig source entity has no position")
        return

    length = params.get("tunnel_length", 1)
    cx, cy = start_pos.x, start_pos.y
    dx, dy = direction
    log.debug("Executing Dig Tunnel", start=(cx, cy), dir=direction, len=length)
    map_changed = False
    source_height_val = (
        gs.game_map.height_map[cy, cx] if gs.game_map.in_bounds(cx, cy) else 0
    )
    source_height = int(source_height_val)

    for i in range(1, length + 1):
        nx, ny = cx + dx * i, cy + dy * i
        if not gs.game_map.in_bounds(nx, ny):
            log.debug("Dig hit map boundary", pos=(nx, ny))
            break
        current_tile = gs.game_map.tiles[ny, nx]
        if current_tile == TILE_ID_WALL:
            gs.game_map.tiles[ny, nx] = TILE_ID_FLOOR
            # Set height/ceiling reasonably based on source
            gs.game_map.height_map[ny, nx] = source_height
            # Example ceiling
            gs.game_map.ceiling_map[ny, nx] = source_height + 6
            log.debug("Dug wall", pos=(nx, ny))
            map_changed = True
        else:
            log.debug("Dig stopped by non-wall tile", pos=(nx, ny), tile=current_tile)
            break
    if map_changed:
        gs.game_map.update_tile_transparency()  # Crucial after changing tiles


def create_portal(context: Dict[str, Any], params: Dict[str, Any]):
    """Creates a portal entity at the target location."""
    gs: "GameState" = context.get("game_state")
    target_pos = context.get("target_pos")
    portal_template_id = params.get("portal_template", "default_portal")
    linked_positions_param = params.get("linked_positions")
    target_map_override = params.get("target_map")

    if gs is None or target_pos is None:
        log.warning("CreatePortal missing context", pos=target_pos)
        return

    template_data = gs.entity_templates.get_template(portal_template_id)
    if not template_data:
        log.error(
            "CreatePortal failed: Unknown template",
            template=portal_template_id,
        )
        return

    portal_glyph = template_data.get("glyph", params.get("glyph", 62))
    portal_color = tuple(template_data.get("color", params.get("color", (255, 0, 255))))
    portal_name = template_data.get("name", params.get("name", "Portal"))
    blocks = template_data.get("blocks_movement", params.get("blocks_movement", False))
    linked_positions = linked_positions_param or template_data.get(
        "linked_positions", []
    )
    target_map = target_map_override or template_data.get("target_map")

    log.info(
        "Creating portal",
        template=portal_template_id,
        pos=target_pos,
        link=linked_positions,
    )

    # Check if tile is blocked
    if not gs.game_map.is_walkable(target_pos[0], target_pos[1]):
        log.warning("Cannot create portal: location not walkable", pos=target_pos)
        return
    if gs.entity_registry.get_blocking_entity_at(target_pos[0], target_pos[1]):
        log.warning("Cannot create portal: location blocked by entity", pos=target_pos)
        return

    new_id = gs.entity_registry.create_entity(
        x=target_pos[0],
        y=target_pos[1],
        glyph=portal_glyph,
        color_fg=portal_color,
        name=portal_name,
        blocks_movement=blocks,
        hp=1,
        max_hp=1,  # Portals likely indestructible by HP
    )
    if new_id is not None:
        log.info("Portal entity created", entity_id=new_id, pos=target_pos)
        gs.entity_registry.set_entity_component(
            new_id, "linked_positions", linked_positions
        )
        gs.entity_registry.set_entity_component(new_id, "target_map", target_map)
    else:
        log.error("Failed to create portal entity", pos=target_pos)


def attempt_spawn_entity(context: Dict[str, Any], params: Dict[str, Any]):
    """Attempts to spawn a specified entity at the target location based on chance."""
    gs: "GameState" = context.get("game_state")
    rng: GameRNG | None = context.get("rng")
    target_pos = context.get("target_pos")
    chance = params.get("chance", 100)
    entity_template_id = params.get("entity_template")  # ID string, e.g., "goblin"

    if gs is None or target_pos is None or not entity_template_id or rng is None:
        log.warning(
            "AttemptSpawn missing context",
            pos=target_pos,
            template=entity_template_id,
            has_rng=bool(rng),
        )
        return

    roll = rng.get_int(1, 100)
    if roll > chance:
        log.debug("AttemptSpawn failed chance roll", chance=chance, roll=roll)
        return

    template_data = gs.entity_templates.get_template(entity_template_id)
    if not template_data:
        log.error(
            "AttemptSpawn failed: Unknown entity template ID",
            template=entity_template_id,
        )
        return

    log.info("Attempting to spawn entity", template=entity_template_id, pos=target_pos)

    # Check if tile is blocked
    if not gs.game_map.is_walkable(target_pos[0], target_pos[1]):
        log.debug("Cannot spawn entity: location not walkable", pos=target_pos)
        return
    if gs.entity_registry.get_blocking_entity_at(target_pos[0], target_pos[1]):
        log.debug("Cannot spawn entity: location blocked by entity", pos=target_pos)
        return

    new_id = gs.entity_registry.create_entity(
        x=target_pos[0],
        y=target_pos[1],
        glyph=template_data.get("glyph", ord("?")),
        color_fg=tuple(template_data.get("color", (255, 255, 255))),
        name=template_data.get("name", entity_template_id),
        blocks_movement=template_data.get("blocks_movement", True),
        species=template_data.get("species"),
        intelligence=template_data.get("intelligence", 1),
        hp=template_data.get("hp", 1),
        max_hp=template_data.get("hp", 1),
    )
    if new_id is not None:
        log.info(
            "Entity spawned successfully",
            entity_id=new_id,
            template=entity_template_id,
            pos=target_pos,
        )
        # Add message? "A goblin appears!"
    else:
        log.error("Failed to spawn entity", template=entity_template_id, pos=target_pos)


# --- Handler Registry (no changes needed here) ---
EFFECT_LOGIC_HANDLERS: Dict[str, callable] = {
    "heal_target": heal_target,
    "modify_resource": modify_resource,
    "recall_ammo": recall_ammo,
    "apply_status": apply_status,
    "apply_status_in_aoe": apply_status_in_aoe,
    "deal_damage": deal_damage,
    "deal_damage_in_aoe": deal_damage_in_aoe,
    "dig_tunnel": dig_tunnel,
    "create_portal": create_portal,
    "attempt_spawn_entity": attempt_spawn_entity,
}

# --- Art/Substance Dispatcher -------------------------------------------------

# The game's upcoming magic system describes effects using a pair of enums:
# ``Art`` and ``Substance``.  For simple prototypes we map those pairs to the
# existing effect handlers above.  When a pair has no mapping we return a
# default no-op function and log a warning so the caller knows the work had no
# effect.


def _no_op_effect(
    context: Dict[str, Any], params: Dict[str, Any] | None = None
) -> None:
    """Fallback effect when no mapping exists."""
    log.debug("No-op effect executed", context=context)


ART_SUBSTANCE_DISPATCHER: Dict[Tuple[Art, Substance], callable] = {
    # Example mappings – these can be expanded as more of the magic system is
    # implemented.
    (Art.CREATE, Substance.WATER): heal_target,
    (Art.DESTROY, Substance.FIRE): deal_damage,
}


def get_art_substance_handler(art: Art, substance: Substance) -> callable:
    """Return an effect handler for the given ``(Art, Substance)`` pair."""
    handler = ART_SUBSTANCE_DISPATCHER.get((art, substance))
    if handler is None:
        log.warning(
            "No effect handler for art/substance pair", art=art, substance=substance
        )
        return _no_op_effect
    return handler
