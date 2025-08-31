"""Tests for the sound system."""
import sys
import os
import tempfile
import yaml
from pathlib import Path

# Ensure project root on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest

from game.systems.sound import SoundManager, SoundEffect, BackgroundMusic


class TestSoundEffect:
    """Test sound effect functionality."""
    
    def test_sound_effect_creation(self):
        """Test creating a sound effect."""
        config = {
            "files": ["test1.ogg", "test2.ogg"],
            "volume": 0.8,
            "random_pitch": 0.1,
            "conditions": {"target": "player"}
        }
        effect = SoundEffect(config, Path("/test"))
        
        assert effect.files == ["test1.ogg", "test2.ogg"]
        assert effect.volume == 0.8
        assert effect.random_pitch == 0.1
        assert effect.conditions == {"target": "player"}
    
    def test_sound_effect_matches_conditions(self):
        """Test sound effect condition matching."""
        config = {
            "files": ["test.ogg"],
            "conditions": {"target": "player", "terrain": ["floor", "stone"]}
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
        assert selected in ["test1.ogg", "test2.ogg"]
        
        # Empty files should return None
        empty_effect = SoundEffect({"files": []}, Path("/test"))
        assert empty_effect.get_random_file() is None


class TestBackgroundMusic:
    """Test background music functionality."""
    
    def test_background_music_creation(self):
        """Test creating background music."""
        config = {
            "files": ["music1.ogg"],
            "volume": 0.6,
            "loop": True,
            "fade_in_time": 2.0,
            "fade_out_time": 3.0,
            "priority": 10,
            "conditions": {"game_state": ["exploring"]}
        }
        music = BackgroundMusic(config, Path("/test"))
        
        assert music.files == ["music1.ogg"]
        assert music.volume == 0.6
        assert music.loop is True
        assert music.fade_in_time == 2.0
        assert music.fade_out_time == 3.0
        assert music.priority == 10
        assert music.conditions == {"game_state": ["exploring"]}
    
    def test_background_music_matches_conditions(self):
        """Test background music condition matching."""
        config = {
            "files": ["music.ogg"],
            "conditions": {
                "game_state": ["exploring"],
                "min_depth": 5
            }
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
                "music_volume": 0.7
            },
            "sound_effects": {
                "test_effect": {
                    "files": ["test.ogg"],
                    "volume": 0.5,
                    "conditions": {"target": "player"}
                },
                "combat_hit": {
                    "files": ["hit1.ogg", "hit2.ogg"],
                    "volume": 0.8,
                    "random_pitch": 0.2
                }
            },
            "background_music": {
                "exploration": {
                    "files": ["explore.ogg"],
                    "volume": 0.4,
                    "loop": True,
                    "conditions": {"game_state": ["exploring"]}
                },
                "combat": {
                    "files": ["combat.ogg"],
                    "volume": 0.6,
                    "priority": 10,
                    "conditions": {"game_state": ["combat"]}
                }
            },
            "event_mappings": {
                "player_move": "test_effect",
                "deal_damage": "combat_hit"
            }
        }
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
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
            
            # Check loaded background music
            assert "exploration" in manager.background_music
            assert "combat" in manager.background_music
            
            # Check event mappings
            assert manager.event_mappings["player_move"] == "test_effect"
            assert manager.event_mappings["deal_damage"] == "combat_hit"
            
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
            
            # Test combat music (higher priority)
            context = {"game_state": "combat"}
            manager.update_background_music(context)
            assert manager.current_music_name == "combat"
            
            # Test back to exploration
            context = {"game_state": "exploring"}
            manager.update_background_music(context)
            assert manager.current_music_name == "exploration"
            
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
            
            # Test event with no mapping
            manager.handle_game_event("unknown_event", {})
            
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


if __name__ == "__main__":
    pytest.main([__file__])