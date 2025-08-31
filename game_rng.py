"""
Minimal GameRNG implementation for basicrl project.
This is a stub implementation to enable testing and validation of other components.
"""
import json
import random
import uuid
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np


class GameRNG:
    """
    Minimal GameRNG implementation providing the interface expected by the basicrl project.
    Uses Python's random module internally for simplicity.
    """
    
    def __init__(self, seed: Optional[int] = None, metrics: bool = False, buffer_size: int = 10000, max_buffer_size: int = 100000, generator_type: str = "numpy", noise_seed: Optional[int] = None):
        """Initialize GameRNG with given parameters."""
        self.initial_seed = seed if seed is not None else random.randint(0, 2**32-1)
        self.noise_seed = noise_seed if noise_seed is not None else self.initial_seed
        self.generator_type = generator_type
        self.metrics_enabled = metrics
        
        # Initialize internal random state
        self._rng = random.Random(self.initial_seed)
        self._np_rng = np.random.Generator(np.random.PCG64(self.initial_seed))
        
    # Basic integer generation
    def get_int(self, a: int, b: int) -> int:
        """Generate random integer in range [a, b] inclusive."""
        return self._rng.randint(a, b)
    
    def randint(self, a: int, b: int) -> int:
        """Alias for get_int."""
        return self.get_int(a, b)
    
    # Basic float generation  
    def get_float(self, a: float = 0.0, b: float = 1.0) -> float:
        """Generate random float in range [a, b)."""
        return self._rng.uniform(a, b)
    
    def get_floats(self, a: float, b: float, count: int) -> List[float]:
        """Generate multiple random floats in range [a, b)."""
        return [self.get_float(a, b) for _ in range(count)]
    
    def uniform(self, a: float = 0.0, b: float = 1.0) -> float:
        """Alias for get_float."""
        return self.get_float(a, b)
    
    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Generate random number from normal distribution."""
        return self._rng.normalvariate(mu, sigma)
    
    # Sequence operations
    def choice(self, seq: Sequence[Any]) -> Any:
        """Choose random element from sequence."""
        if not seq:
            raise IndexError("Cannot choose from empty sequence")
        return self._rng.choice(seq)
    
    def shuffle(self, seq: List[Any]) -> None:
        """Shuffle sequence in place."""
        self._rng.shuffle(seq)
    
    def sample(self, population: Sequence[Any], k: int) -> List[Any]:
        """Sample k elements from population without replacement."""
        return self._rng.sample(population, k)
    
    # Boolean generation
    def bool(self, probability: float = 0.5) -> bool:
        """Generate random boolean with given probability of True."""
        return self._rng.random() < probability
    
    # Dice rolling
    def dice(self, count: int, sides: int) -> int:
        """Roll dice: count d-sided dice."""
        return sum(self.get_int(1, sides) for _ in range(count))
    
    # UUID generation
    def uuid4(self) -> str:
        """Generate random UUID4 string."""
        return str(uuid.uuid4())
    
    # State management
    def get_state(self) -> Dict[str, Any]:
        """Get current RNG state."""
        return {
            "random_state": self._rng.getstate(),
            "numpy_state": self._np_rng.__getstate__(),
            "initial_seed": self.initial_seed,
            "noise_seed": self.noise_seed
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set RNG state."""
        if "random_state" in state:
            self._rng.setstate(state["random_state"])
        if "numpy_state" in state:
            self._np_rng.__setstate__(state["numpy_state"])
    
    def save_state_to_file(self, filename: str) -> None:
        """Save RNG state to file."""
        state = self.get_state()
        # Convert numpy state to serializable format
        state["numpy_state"] = str(state["numpy_state"])
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state_from_file(self, filename: str) -> None:
        """Load RNG state from file."""
        with open(filename, 'r') as f:
            state = json.load(f)
        # Note: numpy state deserialization is complex, skip for stub
        if "random_state" in state:
            self._rng.setstate(state["random_state"])
    
    # Weighted sampling
    def weighted_sample_ares(self, weights: List[float], k: int = 1) -> List[int]:
        """Weighted sampling using A-Res algorithm (simplified)."""
        if not weights:
            return []
        # Simple weighted choice implementation
        indices = list(range(len(weights)))
        return self._rng.choices(indices, weights=weights, k=k)
    
    def weighted_choice(self, choices: Sequence[Any], weights: Sequence[float]) -> Any:
        """Choose element from choices using weights."""
        if not choices or not weights:
            raise ValueError("Choices and weights cannot be empty")
        if len(choices) != len(weights):
            raise ValueError("Choices and weights must have same length")
        return self._rng.choices(choices, weights=weights, k=1)[0]
    
    # Noise functions (simplified implementations)
    def noise_1d(self, x: float, scale: float = 1.0, seed_offset: int = 0) -> float:
        """1D Perlin noise (simplified)."""
        # Very basic noise implementation
        x_scaled = x * scale + seed_offset
        return np.sin(x_scaled * 12.9898 + 78.233) * 0.5 + 0.5
    
    def noise_2d(self, x: float, y: float, scale: float = 1.0, seed_offset: int = 0) -> float:
        """2D Perlin noise (simplified)."""
        # Very basic noise implementation
        x_scaled = x * scale + seed_offset
        y_scaled = y * scale + seed_offset
        return (np.sin(x_scaled * 12.9898 + y_scaled * 78.233) * 43758.5453) % 1.0 * 2.0 - 1.0
    
    # Metrics (stub)
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {"calls": 0, "enabled": self.metrics_enabled}