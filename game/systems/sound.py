"""Sound system for BasicRL.

This module provides a comprehensive sound effects and background music system
that integrates with the game's event-driven architecture. It supports:

- Situational background music based on game state
- Context-aware sound effects 
- Distance-based audio falloff
- Environmental audio effects
- Volume and audio settings management

The sound system is designed to be non-intrusive and can be disabled entirely
through configuration.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import structlog
import yaml

if TYPE_CHECKING:
    from game.game_state import GameState

log = structlog.get_logger(__name__)

# Audio backend detection - try multiple backends for compatibility
AUDIO_BACKEND = None
try:
    import pygame.mixer as audio_backend
    AUDIO_BACKEND = "pygame"
    log.info("Using pygame audio backend")
except ImportError:
    try:
        import simpleaudio as audio_backend
        AUDIO_BACKEND = "simpleaudio"
        log.info("Using simpleaudio backend")
    except ImportError:
        log.warning("No audio backend available - sound system will be disabled")


class SoundEffect:
    """Represents a single sound effect with its properties."""
    
    def __init__(self, config: Dict[str, Any], base_path: Path):
        self.files = config.get("files", [])
        self.volume = config.get("volume", 1.0)
        self.random_pitch = config.get("random_pitch", 0.0)
        self.conditions = config.get("conditions", {})
        self.base_path = base_path
        self._loaded_sounds = {}
        
    def get_random_file(self) -> Optional[str]:
        """Get a random sound file from the available options."""
        if not self.files:
            return None
        return random.choice(self.files)
    
    def matches_conditions(self, context: Dict[str, Any]) -> bool:
        """Check if this sound effect matches the given context."""
        for condition_key, condition_value in self.conditions.items():
            context_value = context.get(condition_key)
            
            if isinstance(condition_value, list):
                if context_value not in condition_value:
                    return False
            else:
                if context_value != condition_value:
                    return False
        return True


class BackgroundMusic:
    """Represents background music with situational awareness."""
    
    def __init__(self, config: Dict[str, Any], base_path: Path):
        self.files = config.get("files", [])
        self.volume = config.get("volume", 1.0)
        self.loop = config.get("loop", True)
        self.fade_in_time = config.get("fade_in_time", 1.0)
        self.fade_out_time = config.get("fade_out_time", 1.0)
        self.priority = config.get("priority", 0)
        self.conditions = config.get("conditions", {})
        self.base_path = base_path
        self._current_track = None
        
    def get_random_file(self) -> Optional[str]:
        """Get a random music file from the available options."""
        if not self.files:
            return None
        return random.choice(self.files)
    
    def matches_conditions(self, context: Dict[str, Any]) -> bool:
        """Check if this background music matches the current game context."""
        for condition_key, condition_value in self.conditions.items():
            context_value = context.get(condition_key)
            
            if condition_key == "min_depth":
                player_depth = context.get("depth", 0)
                if player_depth < condition_value:
                    return False
            elif isinstance(condition_value, list):
                if context_value not in condition_value:
                    return False
            else:
                if context_value != condition_value:
                    return False
        return True


class SoundManager:
    """Main sound system manager."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.enabled = False
        self.sound_effects: Dict[str, SoundEffect] = {}
        self.background_music: Dict[str, BackgroundMusic] = {}
        self.event_mappings: Dict[str, str] = {}
        self.situational_modifiers: Dict[str, Any] = {}
        self.current_music = None
        self.current_music_name = None
        self.active_sounds: Set[Any] = set()
        
        # Audio settings
        self.master_volume = 1.0
        self.sfx_volume = 1.0
        self.music_volume = 1.0
        self.max_concurrent_sounds = 8
        self.sound_fade_distance = 10
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "sounds.yaml"
        
        try:
            self._load_config(config_path)
            if AUDIO_BACKEND and self.enabled:
                self._initialize_audio_backend()
        except Exception as e:
            log.warning(f"Failed to initialize sound system: {e}")
            self.enabled = False
    
    def _load_config(self, config_path: Path) -> None:
        """Load sound configuration from YAML file."""
        if not config_path.exists():
            log.warning(f"Sound config file not found: {config_path}")
            return
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Load general audio settings
        audio_config = config.get("audio", {})
        self.enabled = audio_config.get("enabled", True) and AUDIO_BACKEND is not None
        self.master_volume = audio_config.get("master_volume", 1.0)
        self.sfx_volume = audio_config.get("sfx_volume", 1.0)
        self.music_volume = audio_config.get("music_volume", 1.0)
        self.max_concurrent_sounds = audio_config.get("max_concurrent_sounds", 8)
        self.sound_fade_distance = audio_config.get("sound_fade_distance", 10)
        
        # Load sound effects
        sfx_config = config.get("sound_effects", {})
        base_sound_path = config_path.parent / "sounds"
        
        for sfx_name, sfx_data in sfx_config.items():
            self.sound_effects[sfx_name] = SoundEffect(sfx_data, base_sound_path)
        
        # Load background music
        music_config = config.get("background_music", {})
        base_music_path = config_path.parent / "music"
        
        for music_name, music_data in music_config.items():
            self.background_music[music_name] = BackgroundMusic(music_data, base_music_path)
        
        # Load event mappings
        self.event_mappings = config.get("event_mappings", {})
        
        # Load situational modifiers
        self.situational_modifiers = config.get("situational_modifiers", {})
        
        log.info(f"Loaded sound config: {len(self.sound_effects)} effects, {len(self.background_music)} music tracks")
    
    def _initialize_audio_backend(self) -> None:
        """Initialize the audio backend."""
        if not self.enabled:
            return
            
        try:
            if AUDIO_BACKEND == "pygame":
                import pygame
                pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
                pygame.mixer.init()
                log.info("Pygame audio backend initialized")
            elif AUDIO_BACKEND == "simpleaudio":
                # simpleaudio doesn't need initialization
                log.info("Simpleaudio backend ready")
        except Exception as e:
            log.error(f"Failed to initialize audio backend: {e}")
            self.enabled = False
    
    def play_sound_effect(self, effect_name: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Play a sound effect with the given context."""
        if not self.enabled or effect_name not in self.sound_effects:
            return False
        
        effect = self.sound_effects[effect_name]
        
        # Check if sound matches context conditions
        if context and not effect.matches_conditions(context):
            return False
        
        # Get sound file
        sound_file = effect.get_random_file()
        if not sound_file:
            return False
        
        # Calculate volume with modifiers
        volume = self._calculate_volume(effect.volume, context)
        
        # Limit concurrent sounds
        if len(self.active_sounds) >= self.max_concurrent_sounds:
            return False
        
        try:
            return self._play_sound_file(sound_file, volume, effect.random_pitch)
        except Exception as e:
            log.warning(f"Failed to play sound effect {effect_name}: {e}")
            return False
    
    def update_background_music(self, context: Dict[str, Any]) -> None:
        """Update background music based on current game context."""
        if not self.enabled:
            return
        
        # Find the best matching music track
        best_music = None
        best_priority = -1
        best_name = None
        
        for music_name, music in self.background_music.items():
            if music.matches_conditions(context) and music.priority > best_priority:
                best_music = music
                best_priority = music.priority
                best_name = music_name
        
        # Switch music if needed
        if best_name != self.current_music_name:
            self._switch_background_music(best_music, best_name, context)
    
    def _switch_background_music(self, new_music: Optional[BackgroundMusic], 
                                music_name: Optional[str], context: Dict[str, Any]) -> None:
        """Switch to new background music with proper fading."""
        if not self.enabled:
            return
            
        # Stop current music
        if self.current_music:
            self._stop_background_music()
        
        # Start new music
        if new_music:
            music_file = new_music.get_random_file()
            if music_file:
                volume = self._calculate_music_volume(new_music.volume, context)
                try:
                    self._play_background_music_file(music_file, volume, new_music.loop)
                    self.current_music_name = music_name
                    log.debug(f"Switched to background music: {music_name}")
                except Exception as e:
                    log.warning(f"Failed to play background music {music_name}: {e}")
    
    def _calculate_volume(self, base_volume: float, context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate final volume with all modifiers applied."""
        final_volume = base_volume * self.sfx_volume * self.master_volume
        
        if not context:
            return max(0.0, min(1.0, final_volume))
        
        # Apply distance-based falloff
        distance = context.get("distance", 0)
        if distance > 0 and self.sound_fade_distance > 0:
            distance_modifier = max(0.0, 1.0 - (distance / self.sound_fade_distance))
            final_volume *= distance_modifier
        
        # Apply environmental modifiers
        environment = context.get("environment")
        if environment and "environment_effects" in self.situational_modifiers:
            env_effects = self.situational_modifiers["environment_effects"].get(environment, {})
            volume_modifier = env_effects.get("volume_modifier", 1.0)
            final_volume *= volume_modifier
        
        return max(0.0, min(1.0, final_volume))
    
    def _calculate_music_volume(self, base_volume: float, context: Dict[str, Any]) -> float:
        """Calculate background music volume with modifiers."""
        return max(0.0, min(1.0, base_volume * self.music_volume * self.master_volume))
    
    def _play_sound_file(self, filename: str, volume: float, pitch_variance: float = 0.0) -> bool:
        """Play a sound file using the current audio backend."""
        # This is a placeholder - actual implementation depends on audio backend
        # and whether sound files exist
        log.debug(f"Would play sound: {filename} at volume {volume:.2f}")
        return True
    
    def _play_background_music_file(self, filename: str, volume: float, loop: bool = True) -> None:
        """Play background music using the current audio backend."""
        # This is a placeholder - actual implementation depends on audio backend
        # and whether music files exist
        log.debug(f"Would play music: {filename} at volume {volume:.2f}, loop={loop}")
    
    def _stop_background_music(self) -> None:
        """Stop the current background music."""
        if self.current_music:
            log.debug(f"Stopping background music: {self.current_music_name}")
            # Actual backend-specific stop code would go here
            self.current_music = None
            self.current_music_name = None
    
    def handle_game_event(self, event_name: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Handle a game event that might trigger sound effects."""
        if not self.enabled:
            return
        
        # Look up sound effect mapping
        sfx_name = self.event_mappings.get(event_name)
        if sfx_name:
            self.play_sound_effect(sfx_name, context)
    
    def set_master_volume(self, volume: float) -> None:
        """Set the master volume (0.0 to 1.0)."""
        self.master_volume = max(0.0, min(1.0, volume))
    
    def set_sfx_volume(self, volume: float) -> None:
        """Set the sound effects volume (0.0 to 1.0)."""
        self.sfx_volume = max(0.0, min(1.0, volume))
    
    def set_music_volume(self, volume: float) -> None:
        """Set the background music volume (0.0 to 1.0)."""
        self.music_volume = max(0.0, min(1.0, volume))
    
    def enable_audio(self, enabled: bool) -> None:
        """Enable or disable the entire audio system."""
        if enabled and not AUDIO_BACKEND:
            log.warning("Cannot enable audio - no backend available")
            return
        
        self.enabled = enabled
        if not enabled and self.current_music:
            self._stop_background_music()
    
    def cleanup(self) -> None:
        """Clean up audio resources."""
        if self.enabled:
            self._stop_background_music()
            self.active_sounds.clear()


# Global sound manager instance
_sound_manager: Optional[SoundManager] = None


def get_sound_manager() -> SoundManager:
    """Get the global sound manager instance."""
    global _sound_manager
    if _sound_manager is None:
        _sound_manager = SoundManager()
    return _sound_manager


def init_sound_system(config_path: Optional[Path] = None) -> SoundManager:
    """Initialize the global sound system."""
    global _sound_manager
    _sound_manager = SoundManager(config_path)
    return _sound_manager


def play_sound(effect_name: str, context: Optional[Dict[str, Any]] = None) -> bool:
    """Convenience function to play a sound effect."""
    return get_sound_manager().play_sound_effect(effect_name, context)


def handle_event(event_name: str, context: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function to handle a game event."""
    get_sound_manager().handle_game_event(event_name, context)


def update_music_context(context: Dict[str, Any]) -> None:
    """Convenience function to update background music context."""
    get_sound_manager().update_background_music(context)