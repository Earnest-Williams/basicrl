# Sound System Documentation

The BasicRL sound system provides situationally appropriate sound effects and background music that enhance the gameplay experience. The system is designed to be:

- **Situational**: Background music and sound effects adapt to the current game context
- **Configurable**: All sounds and settings are defined in YAML configuration files
- **Non-intrusive**: Can be completely disabled without affecting gameplay
- **Event-driven**: Integrates seamlessly with the existing effects system

## Quick Start

1. **Enable Audio**: Ensure audio is enabled in `config/sounds.yaml`:
   ```yaml
   audio:
     enabled: true
   ```

2. **Install Audio Backend**: Install one of the supported audio libraries:
   ```bash
   pip install pygame
   # OR
   pip install simpleaudio
   ```

3. **Add Audio Files**: Create directories and add audio files:
   ```
   config/
   ├── sounds/          # Sound effect files (.ogg format)
   │   ├── footstep_stone_01.ogg
   │   ├── sword_hit_01.ogg
   │   └── ...
   └── music/           # Background music files (.ogg format)
       ├── exploration_ambient_01.ogg
       ├── combat_intense_01.ogg
       └── ...
   ```

## Architecture

### Components

- **SoundManager**: Central manager that handles all audio operations
- **SoundEffect**: Individual sound effects with conditions and properties
- **BackgroundMusic**: Situational background music with priority system
- **Event Integration**: Automatic triggering based on game events

### Configuration

All sound configuration is stored in `config/sounds.yaml`:

```yaml
# General audio settings
audio:
  enabled: true
  master_volume: 0.7
  sfx_volume: 0.8
  music_volume: 0.6

# Individual sound effects
sound_effects:
  player_move:
    files: ["footstep_01.ogg", "footstep_02.ogg"]
    volume: 0.5
    conditions:
      terrain: ["floor", "corridor"]

# Background music with situational awareness
background_music:
  combat:
    files: ["combat_music.ogg"]
    volume: 0.6
    priority: 10
    conditions:
      game_state: ["combat"]

# Event to sound effect mappings
event_mappings:
  "heal_target": "player_healed"
  "deal_damage": "player_damaged"
```

## Integration Points

### Effects System

The sound system automatically integrates with the effects system. When effects like `heal_target` or `deal_damage` are executed, appropriate sounds are played based on the context.

```python
# In effect handlers, sounds are triggered automatically:
handle_event("heal_target", {
    "target": "player",
    "amount": 10,
    "visible": True
})
```

### Movement System

Player movement automatically triggers appropriate footstep sounds based on terrain type.

### Game State

Background music updates automatically based on the current game state:
- **Exploration**: Calm ambient music during normal exploration
- **Combat**: Intense music when enemies are nearby
- **Deep Dungeon**: Atmospheric music at greater depths
- **Boss Encounters**: Epic music when facing boss enemies

## Sound Effect Configuration

### Basic Sound Effect

```yaml
sound_name:
  files: ["sound1.ogg", "sound2.ogg"]  # Random selection
  volume: 0.8                          # Base volume (0.0-1.0)
  random_pitch: 0.1                    # ±10% pitch variation
```

### Conditional Sound Effect

```yaml
contextual_sound:
  files: ["sound.ogg"]
  volume: 0.6
  conditions:
    target: "player"                   # Only when targeting player
    terrain: ["floor", "stone"]        # Only on specific terrain
    damage_type: "fire"                # Only for fire damage
```

## Background Music Configuration

### Basic Music Track

```yaml
music_name:
  files: ["music.ogg"]
  volume: 0.5
  loop: true
  fade_in_time: 2.0
  fade_out_time: 3.0
```

### Situational Music

```yaml
combat_music:
  files: ["combat1.ogg", "combat2.ogg"]
  volume: 0.6
  priority: 10                         # Higher priority than exploration
  conditions:
    game_state: ["combat"]             # Only during combat
    min_depth: 5                       # Only at depth 5 or deeper
```

## Programmatic Usage

### Playing Sound Effects

```python
from game.systems.sound import play_sound

# Play a sound effect directly
play_sound("spell_cast", {
    "spell_type": "fireball",
    "visible": True
})
```

### Handling Game Events

```python
from game.systems.sound import handle_event

# Trigger event-mapped sounds
handle_event("item_pickup", {
    "item_type": "potion",
    "value": "high"
})
```

### Updating Music Context

```python
from game.systems.sound import update_music_context

# Update background music based on game state
update_music_context({
    "game_state": "combat",
    "depth": 10,
    "enemy_type": ["boss"],
    "player_hp_percent": 0.3
})
```

### Volume Controls

```python
from game.systems.sound import get_sound_manager

sound_manager = get_sound_manager()
sound_manager.set_master_volume(0.8)
sound_manager.set_sfx_volume(0.7)
sound_manager.set_music_volume(0.5)
```

## Environmental Effects

The sound system supports environmental modifications:

```yaml
situational_modifiers:
  # Distance-based volume falloff
  distance_falloff:
    enabled: true
    max_distance: 15
    
  # Environmental acoustics
  environment_effects:
    cavern:
      reverb: 0.3
      volume_modifier: 1.1
    water:
      low_pass_filter: 0.7
      volume_modifier: 0.8
```

## Event Mappings

Common game events are automatically mapped to sound effects:

| Event | Default Sound Effect | Context |
|-------|---------------------|---------|
| `heal_target` | `player_healed` | Healing effects |
| `deal_damage` | `player_damaged` | Damage effects |
| `player_move` | `player_move` | Player movement |
| `recall_ammo` | `teleport` | Magical item return |
| `level_up` | `level_up` | Character progression |

## File Format Support

The sound system expects audio files in OGG Vorbis format (`.ogg`) for best compatibility. Files should be placed in:

- `config/sounds/` - Sound effects
- `config/music/` - Background music

## Performance Considerations

- **Concurrent Sounds**: Limited by `max_concurrent_sounds` setting
- **Memory Usage**: Sounds are loaded on-demand
- **CPU Impact**: Minimal impact when audio is disabled
- **File Size**: Use compressed OGG format for smaller files

## Debugging

Enable debug logging to see sound system activity:

```python
import structlog
structlog.configure(level="DEBUG")
```

This will show:
- Sound effect loading and playback
- Background music transitions
- Event handling
- Condition matching

## Troubleshooting

### No Sound Playback

1. Check that audio is enabled in `config/sounds.yaml`
2. Verify audio backend is installed (`pygame` or `simpleaudio`)
3. Ensure audio files exist in correct directories
4. Check volume settings are not zero
5. Verify file format is OGG Vorbis

### Wrong Sounds Playing

1. Check event mappings in `config/sounds.yaml`
2. Verify sound effect conditions match your context
3. Review background music priorities and conditions

### Performance Issues

1. Reduce `max_concurrent_sounds` setting
2. Use lower quality audio files
3. Disable environmental effects if not needed
4. Consider disabling audio entirely for better performance

## Integration with Mods

The sound system is fully compatible with the modding system:

1. **Custom Sounds**: Add new sound effects to `config/sounds.yaml`
2. **Custom Music**: Add new background music tracks
3. **Event Mapping**: Map custom events to sound effects
4. **Programmatic Control**: Use the sound API in custom code

## Future Enhancements

Planned features for future versions:

- **3D Positional Audio**: Distance and direction-based audio
- **Dynamic Music**: Adaptive music that changes based on gameplay
- **Voice Acting**: Support for character voices and narration
- **Audio Compression**: Runtime audio compression for smaller file sizes
- **Platform Audio**: Integration with platform-specific audio systems