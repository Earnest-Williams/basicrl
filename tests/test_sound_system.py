"""Tests for the sound system."""

import os
import tempfile
import yaml
from pathlib import Path

import pytest
import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine

from game.systems.sound import SoundManager, SoundEffect, BackgroundMusic
from game.world.game_map import GameMap, TILE_ID_FLOOR, TILE_ID_WALL
from game.constants import FeatureType, FlowType
from pathfinding.perception_systems import compute_noise_map


class TestSoundEffect:
    """Test sound effect functionality."""

    def test_sound_effect_creation(self):
        """Test creating a sound effect."""
        config = {
            "files": ["test1.ogg", "test2.ogg"],
            "volume": 0.8,
            "random_pitch": 0.1,
            "conditions": {"target": "player"},
        }
        effect = SoundEffect(config, Path("/test"))

        assert effect.files == ["test1.ogg", "test2.ogg"]
        assert effect.volume == 0.8
        assert effect.random_pitch == 0.1
        assert effect.conditions == {"target": "player"}

    def test_procedural_sound_effect_creation(self):
        """SoundEffect should support procedural generators."""
        config = {
            "type": "procedural",
            "generator": "footsteps",
            "volume": 0.5,
            "settings": {"duration": 0.1},
        }
        effect = SoundEffect(config, Path("/test"))

        assert effect.effect_type == "procedural"
        assert effect.generator == "footsteps"
        assert effect.settings == {"duration": 0.1}

    def test_sound_effect_matches_conditions(self):
        """Test sound effect condition matching."""
        config = {
            "files": ["test.ogg"],
            "conditions": {"target": "player", "terrain": ["floor", "stone"]},
        }
        effect = SoundEffect(config, Path("/test"))

        # Should match
        context1 = {"target": "player", "terrain": "floor"}
        assert effect.matches_conditions(context1)

        # Should match
        context2 = {"target": "player", "terrain": "stone"}
        assert effect.matches_conditions(context2)

        # Should not match - wrong target
        context3 = {"target": "enemy", "terrain": "floor"}
        assert not effect.matches_conditions(context3)

        # Should not match - wrong terrain
        context4 = {"target": "player", "terrain": "water"}
        assert not effect.matches_conditions(context4)

    def test_sound_effect_random_file(self):
        """Test getting random sound file."""
        config = {"files": ["test1.ogg", "test2.ogg"]}
        effect = SoundEffect(config, Path("/test"))

        # Should return one of the files
        selected = effect.get_random_file()
        assert selected.name in ["test1.ogg", "test2.ogg"]

        # Empty files should return None
        empty_effect = SoundEffect({"files": []}, Path("/test"))
        assert empty_effect.get_random_file() is None


class TestBackgroundMusic:
    """Test background music functionality."""

    def test_background_music_creation(self):
        """Test creating background music."""
        config = {
            "generator": {"tempo": 90, "harmony": "major", "intensity": 0.3},
            "volume": 0.6,
            "loop": True,
            "fade_in_time": 2.0,
            "fade_out_time": 3.0,
            "priority": 10,
            "conditions": {"game_state": ["exploring"]},
        }
        music = BackgroundMusic(config, Path("/test"))
        assert music.generator_settings == {
            "tempo": 90,
            "harmony": "major",
            "intensity": 0.3,
        }
        assert music.volume == 0.6
        assert music.loop is True
        assert music.fade_in_time == 2.0
        assert music.fade_out_time == 3.0
        assert music.priority == 10
        assert music.conditions == {"game_state": ["exploring"]}

    def test_background_music_matches_conditions(self):
        """Test background music condition matching."""
        config = {
            "generator": {"tempo": 100, "harmony": "minor", "intensity": 0.5},
            "conditions": {"game_state": ["exploring"], "min_depth": 5},
        }
        music = BackgroundMusic(config, Path("/test"))

        # Should match
        context1 = {"game_state": "exploring", "depth": 10}
        assert music.matches_conditions(context1)

        # Should not match - wrong game state
        context2 = {"game_state": "combat", "depth": 10}
        assert not music.matches_conditions(context2)

        # Should not match - depth too low
        context3 = {"game_state": "exploring", "depth": 3}
        assert not music.matches_conditions(context3)


class TestSoundManager:
    """Test sound manager functionality."""

    def create_test_config(self) -> Path:
        """Create a temporary sound configuration file for testing."""
        config = {
            "audio": {
                "enabled": False,  # Disable actual audio for testing
                "master_volume": 0.8,
                "sfx_volume": 0.9,
                "music_volume": 0.7,
            },
            "sound_effects": {
                "test_effect": {
                    "files": ["test.ogg"],
                    "volume": 0.5,
                    "conditions": {"target": "player"},
                },
                "combat_hit": {
                    "files": ["hit1.ogg", "hit2.ogg"],
                    "volume": 0.8,
                    "random_pitch": 0.2,
                },
                "magic_proc": {
                    "type": "procedural",
                    "generator": "magic",
                    "volume": 0.6,
                    "settings": {"duration": 0.1},
                },
            },
            "background_music": {
                "exploration": {
                    "generator": {"tempo": 90, "harmony": "major", "intensity": 0.3},
                    "volume": 0.4,
                    "loop": True,
                    "conditions": {"game_state": ["exploring"]},
                },
                "combat": {
                    "generator": {"tempo": 140, "harmony": "minor", "intensity": 0.8},
                    "volume": 0.6,
                    "priority": 10,
                    "conditions": {"game_state": ["combat"]},
                },
                "deep_dungeon": {
                    "generator": {"tempo": 70, "harmony": "minor", "intensity": 0.6},
                    "volume": 0.5,
                    "priority": 5,
                    "conditions": {"min_depth": 10, "game_state": ["exploring"]},
                },
                "night": {
                    "generator": {"tempo": 80, "harmony": "minor", "intensity": 0.4},
                    "volume": 0.4,
                    "priority": 3,
                    "conditions": {
                        "time_of_day": ["night"],
                        "game_state": ["exploring"],
                    },
                },
            },
            "event_mappings": {
                "player_move": "test_effect",
                "deal_damage": "combat_hit",
                "cast_spell": "magic_proc",
            },
            "situational_modifiers": {
                "environment_effects": {
                    "cavern": {
                        "reverb": 0.4,
                        "volume_modifier": 1.0,
                    },
                    "surface": {
                        "low_pass_filter": 0.6,
                        "volume_modifier": 0.9,
                    },
                },
                "time_of_day": {
                    "day": {"volume_modifier": 1.0, "low_pass_modifier": 1.0},
                    "night": {
                        "volume_modifier": 0.8,
                        "reverb_modifier": 1.2,
                        "low_pass_modifier": 1.1,
                    },
                },
                "occlusion": {
                    "wall_absorption": 0.5,
                    "rear_attenuation": 0.5,
                },
            },
        }

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            return Path(f.name)

    def test_sound_manager_initialization(self):
        """Test sound manager initialization."""
        config_path = self.create_test_config()
        try:
            manager = SoundManager(config_path)

            # Should be disabled due to config
            assert not manager.enabled
            assert manager.master_volume == 0.8
            assert manager.sfx_volume == 0.9
            assert manager.music_volume == 0.7

            # Check loaded sound effects
            assert "test_effect" in manager.sound_effects
            assert "combat_hit" in manager.sound_effects
            assert "magic_proc" in manager.sound_effects

            # Check loaded background music
            assert "exploration" in manager.background_music
            assert "combat" in manager.background_music
            assert "deep_dungeon" in manager.background_music
            assert "night" in manager.background_music

            # Check event mappings
            assert manager.event_mappings["player_move"] == "test_effect"
            assert manager.event_mappings["deal_damage"] == "combat_hit"
            assert manager.event_mappings["cast_spell"] == "magic_proc"

        finally:
            os.unlink(config_path)

    def test_calculate_volume_direction_and_occlusion(self):
        """Volume should be reduced for occlusion and sounds behind listener."""
        config_path = self.create_test_config()
        try:
            manager = SoundManager(config_path)

            gm = GameMap(width=3, height=1)
            gm.tiles[0, :] = TILE_ID_FLOOR
            gm.update_tile_transparency()

            front = manager._calculate_volume(
                1.0,
                {},
                source_pos=(2, 0),
                listener_pos=(1, 0, 0.0),
                listener_orientation=(1.0, 0.0),
                game_map=gm,
            )
            back = manager._calculate_volume(
                1.0,
                {},
                source_pos=(0, 0),
                listener_pos=(1, 0, 0.0),
                listener_orientation=(1.0, 0.0),
                game_map=gm,
            )
            assert back < front

            gm2 = GameMap(width=3, height=1)
            gm2.tiles[0, :] = TILE_ID_FLOOR
            gm2.tiles[0, 1] = TILE_ID_WALL
            gm2.update_tile_transparency()
            occluded = manager._calculate_volume(
                1.0,
                {},
                source_pos=(2, 0),
                listener_pos=(0, 0, 0.0),
                listener_orientation=(1.0, 0.0),
                game_map=gm2,
            )
            assert occluded < front
        finally:
            os.unlink(config_path)

    def test_sound_manager_play_effect(self):
        """Test playing sound effects."""
        config_path = self.create_test_config()
        try:
            manager = SoundManager(config_path)
            manager.enabled = True  # Enable for testing

            # Should succeed (but not actually play due to mock backend)
            result = manager.play_sound_effect("test_effect", {"target": "player"})
            assert result is True

            # Procedural effect should also succeed
            result = manager.play_sound_effect("magic_proc", {})
            assert result is True

            # Should fail due to condition mismatch
            result = manager.play_sound_effect("test_effect", {"target": "enemy"})
            assert result is False

            # Should fail due to non-existent effect
            result = manager.play_sound_effect("nonexistent", {})
            assert result is False

        finally:
            os.unlink(config_path)

    def test_sound_manager_background_music(self):
        """Test background music management."""
        config_path = self.create_test_config()
        try:
            manager = SoundManager(config_path)
            manager.enabled = True  # Enable for testing

            # Test exploration music
            context = {"game_state": "exploring"}
            manager.update_background_music(context)
            assert manager.current_music_name == "exploration"
            assert manager.current_music_file is not None

            # Test combat music (higher priority)
            context = {"game_state": "combat"}
            manager.update_background_music(context)
            assert manager.current_music_name == "combat"

            # Test depth-based music
            context = {"game_state": "exploring", "depth": 12}
            manager.update_background_music(context)
            assert manager.current_music_name == "deep_dungeon"

            # Test time-of-day music
            context = {"game_state": "exploring", "time_of_day": "night"}
            manager.update_background_music(context)
            assert manager.current_music_name == "night"

        finally:
            os.unlink(config_path)

    def test_sound_manager_event_handling(self):
        """Test game event handling."""
        config_path = self.create_test_config()
        try:
            manager = SoundManager(config_path)
            manager.enabled = True  # Enable for testing

            # Test event that maps to sound effect
            # This should not raise an exception
            manager.handle_game_event("player_move", {"target": "player"})
            manager.handle_game_event("cast_spell", {})

            # Test event with no mapping
            manager.handle_game_event("unknown_event", {})

        finally:
            os.unlink(config_path)

    def test_environment_effects_change_between_biomes(self):
        """Effects should change with environment and time of day."""
        config_path = self.create_test_config()
        try:
            manager = SoundManager(config_path)
            manager.enabled = True

            tone = Sine(440).to_audio_segment(duration=200)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tone.export(f.name, format="wav")
                source = Path(f.name)

            try:
                cav_day_path = manager._apply_environment_effects(
                    source, {"environment": "cavern", "time_of_day": "day"}
                )
                cav_night_path = manager._apply_environment_effects(
                    source, {"environment": "cavern", "time_of_day": "night"}
                )
                surface_path = manager._apply_environment_effects(
                    source, {"environment": "surface", "time_of_day": "day"}
                )

                cav_day = AudioSegment.from_wav(cav_day_path)
                cav_night = AudioSegment.from_wav(cav_night_path)
                surface = AudioSegment.from_wav(surface_path)

                assert cav_day.raw_data != cav_night.raw_data
                assert cav_day.raw_data != surface.raw_data
            finally:
                os.unlink(source)
                os.unlink(cav_day_path)
                os.unlink(cav_night_path)
                os.unlink(surface_path)
        finally:
            os.unlink(config_path)

    def test_sound_manager_volume_controls(self):
        """Test volume control methods."""
        config_path = self.create_test_config()
        try:
            manager = SoundManager(config_path)

            # Test volume setters
            manager.set_master_volume(0.5)
            assert manager.master_volume == 0.5

            manager.set_sfx_volume(0.3)
            assert manager.sfx_volume == 0.3

            manager.set_music_volume(0.8)
            assert manager.music_volume == 0.8

            # Test bounds checking
            manager.set_master_volume(1.5)  # Should clamp to 1.0
            assert manager.master_volume == 1.0

            manager.set_master_volume(-0.1)  # Should clamp to 0.0
            assert manager.master_volume == 0.0

        finally:
            os.unlink(config_path)


class TestNoiseAttenuation:
    """Ensure noise maps influence volume through obstacles."""

    def _create_manager(self) -> SoundManager:
        config = {"audio": {"enabled": False}, "sound_effects": {}}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(config, f)
            path = Path(f.name)
        try:
            return SoundManager(path)
        finally:
            os.unlink(path)

    def test_volume_reduced_by_door_and_wall(self):
        manager = self._create_manager()

        terrain = np.full((3, 5), FeatureType.FLOOR, dtype=np.int32)
        source = (0, 1)
        listener = (4, 1, 0.0)

        open_noise = compute_noise_map(
            terrain, 1, 0, terrain.shape, FlowType.REAL_NOISE
        )
        vol_open = manager._calculate_volume(
            1.0, source_pos=source, listener_pos=listener, noise_map=open_noise
        )

        door_map = terrain.copy()
        door_map[:, 2] = FeatureType.WALL
        door_map[1, 2] = FeatureType.CLOSED_DOOR
        door_noise = compute_noise_map(
            door_map, 1, 0, door_map.shape, FlowType.REAL_NOISE
        )
        vol_door = manager._calculate_volume(
            1.0, source_pos=source, listener_pos=listener, noise_map=door_noise
        )

        wall_map = terrain.copy()
        wall_map[:, 2] = FeatureType.WALL
        wall_noise = compute_noise_map(
            wall_map, 1, 0, wall_map.shape, FlowType.REAL_NOISE
        )
        vol_wall = manager._calculate_volume(
            1.0, source_pos=source, listener_pos=listener, noise_map=wall_noise
        )

        assert vol_wall == 0.0
        assert vol_door < vol_open


if __name__ == "__main__":
    pytest.main([__file__])
