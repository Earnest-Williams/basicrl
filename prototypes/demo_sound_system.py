#!/usr/bin/env python3
"""
Sound System Demonstration Script

This script demonstrates the functionality of the BasicRL sound system,
including sound effects and situational background music.
"""

from game.systems.sound import (
    init_sound_system,
    play_sound,
    handle_event,
    update_music_context,
)
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def main():
    """Demonstrate the sound system functionality."""
    print("BasicRL Sound System Demonstration")
    print("=" * 40)

    # Initialize the sound system
    print("\n1. Initializing Sound System...")
    sound_manager = init_sound_system()

    if not sound_manager.enabled:
        print("   ⚠️  Sound system is disabled (no audio backend or disabled in config)")
        print("   💡 This is normal in headless environments")
    else:
        print("   ✅ Sound system initialized successfully")

    print(f"   📊 Loaded {len(sound_manager.sound_effects)} sound effects")
    print(f"   🎵 Loaded {len(sound_manager.background_music)} music tracks")
    print(f"   🔗 Loaded {len(sound_manager.event_mappings)} event mappings")

    # Demonstrate sound effects
    print("\n2. Testing Sound Effects...")

    # Test healing sound
    print("   🩹 Playing healing sound...")
    result = play_sound("player_healed", {"target": "player", "visible": True})
    print(f"      Result: {'✅ Played' if result else '❌ Failed/Disabled'}")

    # Test damage sound
    print("   ⚔️  Playing damage sound...")
    result = play_sound(
        "player_damaged", {"target": "player", "damage_type": "physical"}
    )
    print(f"      Result: {'✅ Played' if result else '❌ Failed/Disabled'}")

    # Test movement sound
    print("   👟 Playing movement sound...")
    result = play_sound("player_move", {"terrain": "floor", "entity": "player"})
    print(f"      Result: {'✅ Played' if result else '❌ Failed/Disabled'}")

    # Demonstrate event handling
    print("\n3. Testing Game Event Handling...")

    # Test heal event
    print("   🔄 Triggering heal event...")
    handle_event("heal_target", {"target": "player", "amount": 10})
    print("      Event processed (would play healing sound)")

    # Test damage event
    print("   🔄 Triggering damage event...")
    handle_event(
        "deal_damage", {"target": "player", "damage_type": "fire", "amount": 5}
    )
    print("      Event processed (would play damage sound)")

    # Test recall ammo event
    print("   🔄 Triggering recall ammo event...")
    handle_event("recall_ammo", {"item_type": "projectile", "target": "player"})
    print("      Event processed (would play teleport sound)")

    # Demonstrate situational background music
    print("\n4. Testing Situational Background Music...")

    # Test exploration music
    print("   🎶 Setting exploration context...")
    exploration_context = {
        "game_state": "exploring",
        "depth": 1,
        "turn": 10,
        "player_hp_percent": 1.0,
    }
    update_music_context(exploration_context)
    current_music = sound_manager.current_music_name
    print(f"      Current music: {current_music or 'None'}")

    # Test combat music
    print("   ⚔️  Setting combat context...")
    combat_context = {
        "game_state": "combat",
        "depth": 3,
        "turn": 50,
        "player_hp_percent": 0.7,
    }
    update_music_context(combat_context)
    current_music = sound_manager.current_music_name
    print(f"      Current music: {current_music or 'None'}")

    # Test deep dungeon music
    print("   🕳️  Setting deep dungeon context...")
    deep_context = {
        "game_state": "exploring",
        "depth": 15,
        "turn": 200,
        "player_hp_percent": 0.8,
    }
    update_music_context(deep_context)
    current_music = sound_manager.current_music_name
    print(f"      Current music: {current_music or 'None'}")

    # Test boss encounter music
    print("   👹 Setting boss encounter context...")
    boss_context = {
        "game_state": "combat",
        "depth": 10,
        "enemy_type": ["boss"],
        "player_hp_percent": 0.5,
    }
    update_music_context(boss_context)
    current_music = sound_manager.current_music_name
    print(f"      Current music: {current_music or 'None'}")

    # Demonstrate volume controls
    print("\n5. Testing Volume Controls...")
    print("   🔊 Testing volume controls...")

    original_master = sound_manager.master_volume
    original_sfx = sound_manager.sfx_volume
    original_music = sound_manager.music_volume

    print(
        f"      Original volumes - Master: {original_master}, SFX: {original_sfx}, Music: {original_music}"
    )

    sound_manager.set_master_volume(0.5)
    sound_manager.set_sfx_volume(0.3)
    sound_manager.set_music_volume(0.8)

    print(
        f"      New volumes - Master: {sound_manager.master_volume}, SFX: {sound_manager.sfx_volume}, Music: {sound_manager.music_volume}"
    )

    # Restore original volumes
    sound_manager.set_master_volume(original_master)
    sound_manager.set_sfx_volume(original_sfx)
    sound_manager.set_music_volume(original_music)

    print("      Volumes restored")

    # Show configuration summary
    print("\n6. Configuration Summary...")
    print("   📋 Available Sound Effects:")
    for name in sorted(sound_manager.sound_effects.keys()):
        effect = sound_manager.sound_effects[name]
        conditions = list(effect.conditions.keys()) if effect.conditions else ["none"]
        print(f"      • {name} (conditions: {', '.join(conditions)})")

    print("\n   🎵 Available Background Music:")
    for name in sorted(sound_manager.background_music.keys()):
        music = sound_manager.background_music[name]
        priority = music.priority
        conditions = list(music.conditions.keys()) if music.conditions else ["none"]
        print(
            f"      • {name} (priority: {priority}, conditions: {', '.join(conditions)})"
        )

    print("\n   🔗 Event Mappings:")
    for event, sound in sorted(sound_manager.event_mappings.items()):
        if sound:
            print(f"      • {event} → {sound}")

    print("\n✅ Sound system demonstration complete!")
    print("\nTo add actual audio files:")
    print("   1. Create 'config/sounds/' directory for sound effects")
    print("   2. Create 'config/music/' directory for background music")
    print("   3. Add .ogg audio files matching the names in sounds.yaml")
    print("   4. Install pygame or simpleaudio for audio playback")


if __name__ == "__main__":
    main()
