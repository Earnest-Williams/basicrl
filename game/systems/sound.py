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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np

import structlog
import yaml

from game.world import line_of_sight
from pathfinding.perception_systems import BASE_FLOW_CENTER, NOISE_STRENGTH

if TYPE_CHECKING:
    from game.world.game_map import GameMap

log = structlog.get_logger(__name__)

# Audio backend detection - try multiple backends for compatibility
AUDIO_BACKEND = None
try:
    from openal import oalOpen, Listener, AL_PLAYING
    AUDIO_BACKEND = "pyopenal"
    log.info("Using pyopenal audio backend")
except Exception:  # pragma: no cover - backend availability depends on environment
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
            log.warning(
                "No audio backend available - sound system will be disabled")


class SoundEffect:
    """Represents a single sound effect with its properties."""

    def __init__(self, config: Dict[str, Any], base_path: Path):
        self.effect_type = config.get("type", "file")
        self.files = config.get("files", [])
        self.generator = config.get("generator")
        self.settings = config.get("settings", {})
        self.volume = config.get("volume", 1.0)
        self.random_pitch = config.get("random_pitch", 0.0)
        self.conditions = config.get("conditions", {})
        self.base_path = base_path
        self._loaded_sounds = {}

    def get_random_file(self) -> Optional[Path]:
        """Get a random sound file from the available options."""
        if not self.files:
            return None
        return self.base_path / random.choice(self.files)

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
        # ``generator`` holds parameters for :class:`MusicGenerator`
        self.generator_settings = config.get("generator", {})
        self.volume = config.get("volume", 1.0)
        self.loop = config.get("loop", True)
        self.fade_in_time = config.get("fade_in_time", 1.0)
        self.fade_out_time = config.get("fade_out_time", 1.0)
        self.priority = config.get("priority", 0)
        self.conditions = config.get("conditions", {})
        self.base_path = base_path
        self._current_track = None

    def generate(self, context: Dict[str, Any]) -> Optional[Path]:
        """Generate background music using configured parameters.

        ``context`` may override tempo, harmony or intensity if those keys are
        present.  The function returns a temporary WAV file or ``None`` on
        failure.
        """
        settings = dict(self.generator_settings)
        for key in ("tempo", "harmony", "intensity"):
            if key in context:
                settings[key] = context[key]
        try:
            from game.audio.music import MusicGenerator

            generator = MusicGenerator()
            return generator.generate(**settings)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(f"Failed to generate music: {exc}")
            return None

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
        self.current_music_file: Optional[Path] = None
        self.active_sounds: Set[Any] = set()
        self.listener_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.listener_orientation: Tuple[float, float] = (0.0, 1.0)

        # Audio settings
        self.master_volume = 1.0
        self.sfx_volume = 1.0
        self.music_volume = 1.0
        self.max_concurrent_sounds = 8
        self.sound_fade_distance = 10

        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / \
                "config" / "sounds.yaml"

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
        self.enabled = audio_config.get(
            "enabled", True) and AUDIO_BACKEND is not None
        self.master_volume = audio_config.get("master_volume", 1.0)
        self.sfx_volume = audio_config.get("sfx_volume", 1.0)
        self.music_volume = audio_config.get("music_volume", 1.0)
        self.max_concurrent_sounds = audio_config.get(
            "max_concurrent_sounds", 8)
        self.sound_fade_distance = audio_config.get("sound_fade_distance", 10)

        # Load sound effects
        sfx_config = config.get("sound_effects", {})
        base_sound_path = config_path.parent / "sounds"

        for sfx_name, sfx_data in sfx_config.items():
            self.sound_effects[sfx_name] = SoundEffect(
                sfx_data, base_sound_path)

        # Load background music
        music_config = config.get("background_music", {})
        base_music_path = config_path.parent / "music"

        for music_name, music_data in music_config.items():
            self.background_music[music_name] = BackgroundMusic(
                music_data, base_music_path)

        # Load event mappings
        self.event_mappings = config.get("event_mappings", {})

        # Load situational modifiers
        self.situational_modifiers = config.get("situational_modifiers", {})

        log.info(f"Loaded sound config: {len(self.sound_effects)} effects, {
                 len(self.background_music)} music tracks")

    def _initialize_audio_backend(self) -> None:
        """Initialize the audio backend."""
        if not self.enabled:
            return

        try:
            if AUDIO_BACKEND == "pyopenal":
                # Listener defaults; position will be updated as needed
                Listener.position = self.listener_position
                log.info("PyOpenAL audio backend initialized")
            elif AUDIO_BACKEND == "pygame":
                import pygame
                pygame.mixer.pre_init(
                    frequency=22050, size=-16, channels=2, buffer=512)
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

        # Get sound file or generate procedural audio
        cleanup_files: List[Path] = []
        if effect.effect_type == "procedural":
            try:
                from game.audio import synthesis
                temp_file = synthesis.generate_sound(
                    effect.generator or "", effect.settings)
                cleanup_files.append(temp_file)
                sound_file = temp_file
            except Exception as exc:
                log.warning(f"Failed to generate procedural sound {
                            effect_name}: {exc}")
                return False
        else:
            sound_file = effect.get_random_file()
        if not sound_file:
            return False

        source_pos: Optional[Tuple[float, float]] = None
        listener_pos: Tuple[float, float, float] = self.listener_position
        listener_orient: Tuple[float, float] = self.listener_orientation
        game_map: Optional["GameMap"] = None
        noise_map = None
        if context:
            source_pos = context.get(
                "source_position") or context.get("position")
            lp = context.get("listener_position")
            if lp:
                listener_pos = (lp[0], lp[1], 0.0) if len(
                    lp) == 2 else tuple(lp)
            lo = context.get("listener_orientation")
            if lo:
                listener_orient = (lo[0], lo[1])
            game_map = context.get("game_map")
            noise_map = context.get("noise_map")
            if source_pos and "distance" not in context:
                sx, sy = source_pos
                lx, ly, lz = listener_pos
                dist = math.hypot(sx - lx, sy - ly)
                context = dict(context)
                context["distance"] = dist

        # Calculate volume with modifiers
        volume = self._calculate_volume(
            effect.volume,
            context,
            source_pos,
            listener_pos,
            listener_orient,
            game_map,
            noise_map,
        )

        # Apply environment DSP effects
        if context:
            processed = self._apply_environment_effects(sound_file, context)
            if processed != sound_file:
                cleanup_files.append(processed)
                sound_file = processed

        self._prune_finished_sounds()

        # Limit concurrent sounds
        if len(self.active_sounds) >= self.max_concurrent_sounds:
            return False

        try:
            played = self._play_sound_file(
                sound_file,
                volume,
                effect.random_pitch,
                source_pos,
                listener_pos,
                listener_orient,
            )
            if not played and AUDIO_BACKEND is None:
                # Treat as success in environments without an audio backend
                played = True
            return played
        except Exception as e:
            log.warning(f"Failed to play sound effect {effect_name}: {e}")
            return False
        finally:
            for temp in cleanup_files:
                try:
                    temp.unlink()
                except Exception:
                    pass

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
        if self.current_music or self.current_music_file:
            self._stop_background_music()

        # Start new music using the generator
        if new_music:
            music_file = new_music.generate(context)
            if music_file:
                volume = self._calculate_music_volume(
                    new_music.volume, context)
                try:
                    self._play_background_music_file(
                        music_file, volume, new_music.loop)
                    self.current_music_name = music_name
                    self.current_music_file = music_file
                    log.debug(f"Switched to background music: {music_name}")
                except Exception as e:
                    log.warning(f"Failed to play background music {
                                music_name}: {e}")
                    self.current_music_file = None

    def _calculate_volume(
        self,
        base_volume: float,
        context: Optional[Dict[str, Any]] = None,
        source_pos: Optional[Tuple[float, float]] = None,
        listener_pos: Optional[Tuple[float, float, float]] = None,
        listener_orientation: Optional[Tuple[float, float]] = None,
        game_map: Optional["GameMap"] = None,
        noise_map: Optional[np.ndarray] = None,
    ) -> float:
        """Calculate final volume with all modifiers applied."""
        final_volume = base_volume * self.sfx_volume * self.master_volume

        if context is None:
            context = {}

        # Apply distance-based falloff if not handled by backend
        distance = 0 if AUDIO_BACKEND == "pyopenal" else context.get(
            "distance", 0)
        if distance > 0 and self.sound_fade_distance > 0:
            distance_modifier = max(
                0.0, 1.0 - (distance / self.sound_fade_distance))
            final_volume *= distance_modifier

        # Apply environmental modifiers
        environment = context.get("environment")
        if environment and "environment_effects" in self.situational_modifiers:
            env_effects = self.situational_modifiers["environment_effects"].get(
                environment, {})
            volume_modifier = env_effects.get("volume_modifier", 1.0)
            final_volume *= volume_modifier

        # Apply time-of-day modifiers
        time_of_day = context.get("time_of_day")
        if time_of_day and "time_of_day" in self.situational_modifiers:
            tod_effects = self.situational_modifiers["time_of_day"].get(
                time_of_day, {})
            final_volume *= tod_effects.get("volume_modifier", 1.0)

        occlusion_cfg = self.situational_modifiers.get("occlusion", {})

        # Apply attenuation based on precomputed noise map
        if noise_map is not None and listener_pos is not None:
            nx = int(listener_pos[0])
            ny = int(listener_pos[1])
            if 0 <= ny < noise_map.shape[0] and 0 <= nx < noise_map.shape[1]:
                cost = noise_map[ny, nx]
                infinity = np.iinfo(noise_map.dtype).max // 2
                if cost < infinity:
                    noise_dist = cost - BASE_FLOW_CENTER
                    if NOISE_STRENGTH > 0:
                        noise_modifier = max(
                            0.0, 1.0 - (noise_dist / NOISE_STRENGTH))
                        final_volume *= noise_modifier
                else:
                    final_volume = 0.0

        # Apply directional attenuation for sounds behind the listener
        if source_pos and listener_pos and listener_orientation:
            dx = source_pos[0] - listener_pos[0]
            dy = source_pos[1] - listener_pos[1]
            dist = math.hypot(dx, dy)
            if dist > 0:
                ox, oy = listener_orientation
                dot = (dx * ox + dy * oy) / dist
                if dot < 0:
                    rear = occlusion_cfg.get("rear_attenuation", 0.5)
                    final_volume *= rear

        # Apply occlusion based on map data
        if source_pos and listener_pos and game_map and occlusion_cfg:
            if not line_of_sight(
                int(listener_pos[0]),
                int(listener_pos[1]),
                int(source_pos[0]),
                int(source_pos[1]),
                game_map.transparent,
            ):
                wall_abs = occlusion_cfg.get("wall_absorption", 0.5)
                final_volume *= 1.0 - wall_abs

        return max(0.0, min(1.0, final_volume))

    def _apply_environment_effects(self, fname: Path, context: Dict[str, Any]) -> Path:
        """Apply environmental DSP effects and return processed file path."""
        env_name = context.get("environment")
        if not env_name or "environment_effects" not in self.situational_modifiers:
            return fname
        env_cfg = self.situational_modifiers["environment_effects"].get(
            env_name, {})
        tod_cfg: Dict[str, Any] = {}
        time_of_day = context.get("time_of_day")
        if time_of_day and "time_of_day" in self.situational_modifiers:
            tod_cfg = self.situational_modifiers["time_of_day"].get(
                time_of_day, {})

        reverb_amt = env_cfg.get("reverb")
        lp_amt = env_cfg.get("low_pass_filter")
        eq_cfg = env_cfg.get("eq")
        if reverb_amt is None and lp_amt is None and not eq_cfg:
            return fname

        from tempfile import NamedTemporaryFile
        from pydub import AudioSegment, effects

        try:
            segment = AudioSegment.from_file(fname)
        except Exception:
            # Fallback for environments without ffmpeg when using wav files
            segment = AudioSegment.from_wav(fname)
        if reverb_amt:
            reverb_amt *= tod_cfg.get("reverb_modifier", 1.0)
            segment = self._add_reverb(segment, reverb_amt)
        if lp_amt:
            lp_amt *= tod_cfg.get("low_pass_modifier", 1.0)
            cutoff = int(2000 + (8000 * (1.0 - max(0.0, min(1.0, lp_amt)))))
            segment = effects.low_pass_filter(segment, cutoff)
        if eq_cfg:
            segment = self._apply_eq(segment, eq_cfg)

        tmp = NamedTemporaryFile(delete=False, suffix=".wav")
        segment.export(tmp.name, format="wav")
        return Path(tmp.name)

    def _add_reverb(self, segment, amount: float):
        """Simple reverb using delayed overlays."""
        delay = int(50 + 150 * amount)
        decay = 0.6 * amount
        echo = segment - 20
        for i in range(1, 4):
            segment = segment.overlay(
                echo, position=delay * i, gain_during_overlay=-decay * i * 10)
        return segment

    def _apply_eq(self, segment, eq_cfg: Dict[str, float]):
        """Very simple two-band EQ for bass and treble adjustments."""
        bass_gain = eq_cfg.get("bass")
        treble_gain = eq_cfg.get("treble")
        if bass_gain is not None:
            bass = segment.low_pass_filter(200).apply_gain(bass_gain)
            high = segment.high_pass_filter(200)
            segment = bass.overlay(high)
        if treble_gain is not None:
            treble = segment.high_pass_filter(4000).apply_gain(treble_gain)
            mid = segment.low_pass_filter(4000)
            segment = mid.overlay(treble)
        return segment

    def _calculate_music_volume(self, base_volume: float, context: Dict[str, Any]) -> float:
        """Calculate background music volume with modifiers."""
        return max(0.0, min(1.0, base_volume * self.music_volume * self.master_volume))

    def _prune_finished_sounds(self) -> None:
        """Remove finished sound handles from active list."""
        if not self.active_sounds:
            return
        if AUDIO_BACKEND == "pyopenal":
            finished = {s for s in self.active_sounds if getattr(
                s, "state", None) != AL_PLAYING}
            for src in finished:
                try:
                    src.stop()
                    src.delete()
                except Exception:
                    pass
            self.active_sounds.difference_update(finished)
        elif AUDIO_BACKEND == "pygame":
            finished = {c for c in self.active_sounds if not c.get_busy()}
            for c in finished:
                try:
                    c.stop()
                except Exception:
                    pass
            self.active_sounds.difference_update(finished)
        elif AUDIO_BACKEND == "simpleaudio":
            finished = {p for p in self.active_sounds if not p.is_playing()}
            for p in finished:
                try:
                    p.stop()
                except Exception:
                    pass
            self.active_sounds.difference_update(finished)

    def _calculate_pan(
        self,
        source_pos: Tuple[float, float],
        listener_pos: Tuple[float, float, float],
        listener_orientation: Tuple[float, float],
    ) -> float:
        """Return stereo pan value (-1.0 left to 1.0 right)."""
        dx = source_pos[0] - listener_pos[0]
        dy = source_pos[1] - listener_pos[1]
        angle_to_source = math.atan2(dy, dx)
        listener_angle = math.atan2(
            listener_orientation[1], listener_orientation[0])
        angle = angle_to_source - listener_angle
        pan = math.sin(angle)
        return max(-1.0, min(1.0, pan))

    def _play_sound_file(
        self,
        filename: Path,
        volume: float,
        pitch_variance: float = 0.0,
        source_pos: Optional[Tuple[float, float]] = None,
        listener_pos: Optional[Tuple[float, float, float]] = None,
        listener_orientation: Optional[Tuple[float, float]] = None,
    ) -> bool:
        """Play a sound file using the current audio backend."""
        fname = str(filename)
        if AUDIO_BACKEND == "pyopenal":
            try:
                sound = oalOpen(fname)
                src = sound.play()
                if pitch_variance:
                    pitch = 1.0 + \
                        random.uniform(-pitch_variance, pitch_variance)
                    src.set_pitch(pitch)
                src.set_gain(volume)
                if source_pos:
                    sx, sy = source_pos
                    src.position = (sx, 0.0, sy)
                if listener_pos:
                    Listener.position = listener_pos
                if listener_orientation:
                    Listener.orientation = (
                        listener_orientation[0],
                        listener_orientation[1],
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    )
                self.active_sounds.add(src)
                return True
            except Exception as e:
                log.warning(f"OpenAL failed to play sound {fname}: {e}")
                return False
        elif AUDIO_BACKEND == "pygame":
            try:
                sound = audio_backend.Sound(fname)
                channel = sound.play()
                if channel:
                    if source_pos and listener_pos and listener_orientation:
                        pan = self._calculate_pan(
                            source_pos, listener_pos, listener_orientation)
                        left = volume * (1 - pan) / 2
                        right = volume * (1 + pan) / 2
                        channel.set_volume(left, right)
                    else:
                        channel.set_volume(volume, volume)
                    self.active_sounds.add(channel)
                return channel is not None
            except Exception as e:
                log.warning(f"Pygame failed to play sound {fname}: {e}")
                return False
        elif AUDIO_BACKEND == "simpleaudio":
            try:
                wave_obj = audio_backend.WaveObject.from_wave_file(fname)
                play_obj = wave_obj.play()
                self.active_sounds.add(play_obj)
                return True
            except Exception as e:
                log.warning(f"Simpleaudio failed to play sound {fname}: {e}")
                return False
        log.debug(f"No audio backend to play sound: {fname}")
        return False

    def _play_background_music_file(self, filename: Path, volume: float, loop: bool = True) -> None:
        """Play background music using the current audio backend."""
        fname = str(filename)
        if AUDIO_BACKEND == "pyopenal":
            try:
                sound = oalOpen(fname)
                src = sound.play()
                src.set_gain(volume)
                src.looping = loop
                self.current_music = src
                self.active_sounds.add(src)
            except Exception as e:
                log.warning(f"OpenAL failed to play music {fname}: {e}")
        elif AUDIO_BACKEND == "pygame":
            try:
                audio_backend.music.load(fname)
                audio_backend.music.set_volume(volume)
                audio_backend.music.play(-1 if loop else 0)
                self.current_music = "pygame"
            except Exception as e:
                log.warning(f"Pygame failed to play music {fname}: {e}")
        elif AUDIO_BACKEND == "simpleaudio":
            try:
                wave_obj = audio_backend.WaveObject.from_wave_file(fname)
                play_obj = wave_obj.play()
                self.current_music = play_obj
                self.active_sounds.add(play_obj)
            except Exception as e:
                log.warning(f"Simpleaudio failed to play music {fname}: {e}")
        else:
            log.debug(f"No audio backend to play music: {fname}")

    def _stop_background_music(self) -> None:
        """Stop the current background music."""
        if self.current_music:
            log.debug(f"Stopping background music: {self.current_music_name}")
            try:
                if AUDIO_BACKEND == "pyopenal":
                    self.current_music.stop()
                    self.current_music.delete()
                elif AUDIO_BACKEND == "pygame":
                    audio_backend.music.stop()
                elif AUDIO_BACKEND == "simpleaudio":
                    self.current_music.stop()
            except Exception:
                pass

        if self.current_music_file:
            try:
                self.current_music_file.unlink()
            except Exception:
                pass
            self.current_music_file = None

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

    def set_listener_position(self, x: float, y: float, z: float = 0.0) -> None:
        """Update the 3D listener position."""
        self.listener_position = (x, y, z)
        if AUDIO_BACKEND == "pyopenal" and self.enabled:
            Listener.position = self.listener_position

    def set_listener_orientation(self, x: float, y: float) -> None:
        """Update the 2D listener orientation vector."""
        length = math.hypot(x, y)
        if length == 0:
            return
        self.listener_orientation = (x / length, y / length)
        if AUDIO_BACKEND == "pyopenal" and self.enabled:
            Listener.orientation = (
                self.listener_orientation[0],
                self.listener_orientation[1],
                0.0,
                0.0,
                0.0,
                1.0,
            )

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
            for src in list(self.active_sounds):
                try:
                    if AUDIO_BACKEND == "pyopenal":
                        src.stop()
                        src.delete()
                    else:
                        src.stop()
                except Exception:
                    pass
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
