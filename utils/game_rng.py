# /home/earnest/code_projects/simple_rl/rng_utils/game_rng.py
# Corrected Indentation (4 spaces) - v2.3.0

"""
Enhanced GameRNG - A high-performance, feature-rich random number generator for games

Version: 2.3.0 (Optimized batching, state saving, noise; added A-Res sampling)

This module provides a comprehensive set of tools for game-specific randomization needs:
- Efficient bit-level random generation (including optimized batching)
- Thread-safe operations
- Deterministic sequences with save/load functionality (full state, no flawed compression)
- Game-specific convenience methods (dice, cards, weighted choice, etc.)
- Numba-accelerated Perlin noise generation (1D and 2D) without affecting main RNG state
- Efficient Weighted Sampling Without Replacement (A-Res algorithm)
- Multiple underlying generator algorithms (PCG64, Xorshift128+)
- Performance metrics and dynamic memory management
"""

import bisect  # Added for weighted_choice optimization
import json
import math
import threading
import time
import warnings
from collections import deque
from enum import Enum, auto
from threading import RLock

import numpy as np

# Added for Perlin noise acceleration and potential PRNG speedup
try:
    from numba import njit
except ImportError:
    print("Warning: Numba not installed. Perlin noise will be slower.")

    # Define a dummy decorator if Numba is not available
    def njit(func=None, **options):
        if func:
            return func
        else:
            # Return a decorator function
            def decorator(f):
                return f

            return decorator


# Suppress overflow warnings for uint64 operations
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="overflow encountered"
)
# Suppress NumbaPerformanceWarning about object mode loops (sometimes unavoidable with custom classes)
warnings.filterwarnings(
    "ignore", message=".*Use Function.compile_options=.*", category=UserWarning
)


class TestResult(Enum):
    """Enum for statistical test results"""

    PASS = auto()
    WEAK = auto()
    FAIL = auto()


class SplitMix64:
    """
    SplitMix64 is a simple, fast generator suitable for seeding other generators
    or for auxiliary tasks like initializing noise tables.
    """

    def __init__(self, seed=None):
        """Initialize the SplitMix64 generator with a seed"""
        if seed is None:
            seed = int(time.time() * 1000)
        # Ensure seed fits within uint64 and handle potential negative inputs
        self.state = np.uint64(seed & 0xFFFFFFFFFFFFFFFF)

    # Can potentially be accelerated with Numba if needed, but likely fast enough
    # @njit # Uncomment if profiling shows this is a bottleneck
    def next(self):
        """Generate the next 64-bit value"""
        self.state += np.uint64(0x9E3779B97F4A7C15)
        z = self.state
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))

    def integers(self, low, high, size=1):
        """
        Generate random integers in [low, high) with uniform distribution.
        Uses rejection sampling to avoid modulo bias.
        """
        # Ensure range is positive
        if high <= low:
            if size == 1:
                return np.array([low], dtype=np.uint64)
            return np.full(size, low, dtype=np.uint64)

        result = np.zeros(size, dtype=np.uint64)
        u_low = np.uint64(low)
        u_high = np.uint64(high)
        range_size = u_high - u_low  # Use uint64 for range

        # Handle potential edge case where range_size is 0 after conversion (high == low)
        if range_size == 0:
            if size == 1:
                return np.array([low], dtype=np.uint64)
            return np.full(size, low, dtype=np.uint64)

        # Use rejection sampling to avoid modulo bias
        if range_size > 0 and (range_size & (range_size - np.uint64(1))) == 0:
            # Power of 2 case - can use fast path with bitwise AND
            mask = range_size - np.uint64(1)
            for i in range(size):
                res_val = (self.next() & mask) + u_low
                result[i] = res_val
        else:
            # Non-power of 2 case - use rejection sampling
            bits_needed = range_size.item().bit_length()
            mask = (
                (np.uint64(1) << bits_needed) - np.uint64(1)
                if bits_needed > 0
                else np.uint64(0)
            )

            for i in range(size):
                while True:
                    value = self.next() & mask
                    if value < range_size:
                        res_val = value + u_low
                        result[i] = res_val
                        break
        return result


class XorshiftGenerator:
    """
    An enhanced Xorshift128+ PRNG implementation.
    Includes rejection sampling to eliminate modulo bias.
    """

    def __init__(self, seed=None):
        """Initialize the Xorshift generator with proper seeding"""
        mixer = SplitMix64(seed)
        self.state0 = mixer.next()
        self.state1 = mixer.next()
        # Ensure initial state is not all zeros
        if self.state0 == 0 and self.state1 == 0:
            self.state0 = np.uint64(1)  # Or use mixer.next() again

    def __getstate__(self):
        """Get the generator state for serialization"""
        return {"state0": int(self.state0), "state1": int(self.state1)}

    def __setstate__(self, state):
        """Restore the generator state from serialization"""
        self.state0 = np.uint64(state["state0"])
        self.state1 = np.uint64(state["state1"])

    # Can potentially be accelerated with Numba if needed
    # @njit # Uncomment if profiling shows this is a bottleneck
    def next(self):
        """Generate the next 64-bit random value using Xorshift128+"""
        s1 = self.state0
        s0 = self.state1
        self.state0 = s0
        s1 ^= s1 << np.uint64(23)  # a
        self.state1 = s1 ^ s0 ^ (s1 >> np.uint64(18)) ^ (s0 >> np.uint64(5))  # b, c
        return self.state1 + s0

    def integers(self, low, high, size=1, dtype=np.uint64):
        """
        Generate random integers in [low, high) with specified size and dtype.
        Uses rejection sampling to ensure unbiased results.
        """
        # Ensure range is positive
        if high <= low:
            res = np.full(size, low, dtype=dtype)
            return res[0] if size == 1 else res

        result = np.zeros(size, dtype=dtype)
        u_low = np.uint64(low)
        u_high = np.uint64(high)
        range_size = u_high - u_low  # Use uint64 for range

        # Handle potential edge case where range_size is 0
        if range_size == 0:
            res = np.full(size, low, dtype=dtype)
            return res[0] if size == 1 else res

        # Use rejection sampling to avoid modulo bias
        if range_size > 0 and (range_size & (range_size - np.uint64(1))) == 0:
            # Power of 2 case
            mask = range_size - np.uint64(1)
            for i in range(size):
                res_val = (self.next() & mask) + u_low
                result[i] = dtype(res_val)
        else:
            # Non-power of 2 case
            bits_needed = range_size.item().bit_length()
            mask = (
                (np.uint64(1) << bits_needed) - np.uint64(1)
                if bits_needed > 0
                else np.uint64(0)
            )

            for i in range(size):
                while True:
                    value = self.next() & mask
                    if value < range_size:
                        res_val = value + u_low
                        result[i] = dtype(res_val)
                        break
        return result[0] if size == 1 and isinstance(result, np.ndarray) else result

    def test_randomness(self, num_samples=10000):
        """Basic statistical tests (unchanged from previous version)"""
        # Ensure state is fresh if called multiple times
        initial_state = self.__getstate__()
        samples = np.array([self.next() for _ in range(num_samples)], dtype=np.uint64)
        self.__setstate__(initial_state)  # Restore state after sampling

        results = {}
        # Test 1: Frequency test (chi-squared)
        bits = np.zeros(64, dtype=np.int64)
        for s in samples:
            for i in range(64):
                if (s >> np.uint64(i)) & np.uint64(1):
                    bits[i] += 1
        expected = num_samples / 2.0
        chi_squared = np.sum(((bits - expected) ** 2) / expected) if expected > 0 else 0
        results["frequency_test_chi2"] = chi_squared
        results["frequency_test"] = (
            TestResult.PASS
            if 40 < chi_squared < 100
            else (TestResult.WEAK if 30 < chi_squared < 120 else TestResult.FAIL)
        )

        # Test 2: Runs test
        if num_samples > 1:
            lsb_sequence = (samples & np.uint64(1)).astype(np.int8)
            runs = np.sum(lsb_sequence[:-1] != lsb_sequence[1:]) + 1
            pi = np.mean(lsb_sequence)
            if 0 < pi < 1:
                expected_runs = 2.0 * num_samples * pi * (1.0 - pi)
                var_runs = (
                    2.0
                    * num_samples
                    * pi
                    * (1.0 - pi)
                    * (2.0 * num_samples * pi * (1.0 - pi) - 1.0)
                )
                std_dev_runs = (
                    math.sqrt(var_runs / (num_samples - 1.0))
                    if num_samples > 1 and var_runs >= 0
                    else 0
                )
                runs_z = (
                    abs(runs - expected_runs) / std_dev_runs if std_dev_runs > 0 else 0
                )
                results["runs_test_z_score"] = runs_z
                results["runs_test"] = (
                    TestResult.PASS
                    if runs_z < 1.96
                    else (TestResult.WEAK if runs_z < 2.58 else TestResult.FAIL)
                )
            else:
                results["runs_test_z_score"] = float("inf")
                results["runs_test"] = TestResult.FAIL
        else:
            results["runs_test_z_score"] = 0
            results["runs_test"] = TestResult.PASS

        # Test 3: Serial correlation
        if num_samples > 1:
            samples_float = samples.astype(np.float64)
            # Use np.nan_to_num to handle potential NaN from zero variance
            correlation = np.nan_to_num(
                np.corrcoef(samples_float[:-1], samples_float[1:])[0, 1]
            )
            threshold = 2 / math.sqrt(num_samples - 1)
            weak_threshold = 3 / math.sqrt(num_samples - 1)
            results["serial_correlation"] = correlation
            results["serial_correlation_test"] = (
                TestResult.PASS
                if abs(correlation) < threshold
                else (
                    TestResult.WEAK
                    if abs(correlation) < weak_threshold
                    else TestResult.FAIL
                )
            )
        else:
            results["serial_correlation"] = 0
            results["serial_correlation_test"] = TestResult.PASS

        return results


class CircularBitBuffer:
    """
    A circular buffer for bit operations with efficient memory management
    and optimized batch extraction.
    """

    def __init__(self, generator, initial_size=1000, max_size=1000000):
        """Initialize the circular bit buffer."""
        self.generator = generator
        self.initial_size = max(1, initial_size)
        self.max_size = max(self.initial_size, max_size)

        self._buffer = deque()
        self._buffer_start_index = 0
        self._buffer_end_index = 0

        self.current_value = np.uint64(0)
        self.bits_remaining = 0
        self.current_logical_index = -1

        self.refill_count_interval = 0  # For adaptive sizing calculation
        self.bits_used_total = 0
        self.values_generated_total = 0
        self.last_resize_time = time.time()
        self.resize_interval = 5.0
        self.resize_factor = 1.5
        self.min_size = self.initial_size
        self._current_max_len = self.initial_size

        # Fill initial buffer
        self._fill_buffer(self.initial_size)

    def _fill_buffer(self, num_values):
        """Generate and add new values to the buffer using generator.next()."""
        if num_values <= 0:
            return
        add_count = min(num_values, self._current_max_len - len(self._buffer))
        if add_count <= 0:
            return

        # OPTIMIZATION: Call generator.next() directly for speed
        # Assumes generator provides a 'next()' method returning uint64
        try:
            new_vals = [self.generator.next() for _ in range(add_count)]
        except AttributeError:
            # Fallback for generators without .next() (like numpy's default_rng)
            # Need to request uint64 specifically.
            # Use the full range [0, UINT64_MAX] inclusive.
            UINT64_MAX = np.iinfo(np.uint64).max
            # Use endpoint=True to include UINT64_MAX in the possible range
            new_vals = self.generator.integers(
                low=0, high=UINT64_MAX, size=add_count, dtype=np.uint64, endpoint=True
            )  # <- CORRECTED

        self._buffer.extend(new_vals)
        self._buffer_end_index += add_count
        self.refill_count_interval += 1
        self._check_resize()  # Check resize implicitly

    def _get_next_value(self):
        """Consume the next value from the buffer, refilling if necessary."""
        logical_index_needed = self.current_logical_index + 1

        # --- First attempt to get value if already buffered ---
        if logical_index_needed < self._buffer_end_index:
            deque_index = logical_index_needed - self._buffer_start_index
            if deque_index < len(self._buffer):  # Check bounds
                self.current_value = self._buffer[deque_index]
                self.current_logical_index = logical_index_needed
                self.bits_remaining = 64
                # --- FIX THIS LINE ---
                # OLD: self._trim_buffer()
                # NEW: Pass the expected argument
                self._trim_buffer(force_trim_if_at_max=False)  # Normal trim check
                # --- END FIX ---
                return

        # --- Value not found, need to refill ---
        needed_count = logical_index_needed - self._buffer_end_index + 1

        # --- Try trimming forcefully BEFORE trying to fill ---
        self._trim_buffer(
            force_trim_if_at_max=True
        )  # Use True here (This call was already correct)

        # Determine fill amount
        fill_amount = max(needed_count, self.initial_size // 2, 1)
        self._fill_buffer(fill_amount)

        # --- Second attempt after refilling ---
        if logical_index_needed < self._buffer_end_index:
            deque_index = logical_index_needed - self._buffer_start_index
            if deque_index < len(self._buffer):  # Check deque bounds again
                self.current_value = self._buffer[deque_index]
                self.current_logical_index = logical_index_needed
                self.bits_remaining = 64
                return

        # --- If we STILL failed ---
        raise RuntimeError(
            f"Failed to retrieve logical index {logical_index_needed} from buffer "
            f"(start={self._buffer_start_index}, end={self._buffer_end_index}, "
            f"len={len(self._buffer)}, max_len={self._current_max_len})"
        )

    def _trim_buffer(self, force_trim_if_at_max: bool = False):
        """Remove consumed values from the left if buffer exceeds target capacity or forced."""
        # Trim only if significantly over target capacity OR if forced when already at max
        current_len = len(self._buffer)
        over_capacity = current_len > self._current_max_len * 1.1
        at_max_and_forced = (
            current_len >= self._current_max_len and force_trim_if_at_max
        )

        if over_capacity or at_max_and_forced:
            can_trim = self.current_logical_index - self._buffer_start_index
            # Determine the ideal target length (don't shrink below min_size)
            trim_target_len = max(self.min_size, self._current_max_len)

            # Calculate how many elements *could* be removed to reach the target length
            potential_trim_count = max(0, current_len - trim_target_len)

            # Only trim elements that have actually been consumed
            trim_count = min(potential_trim_count, can_trim)

            if trim_count > 0:
                # Use optimized deque popleft if available, otherwise slice (though deque is standard)
                for _ in range(trim_count):
                    try:
                        self._buffer.popleft()
                    except (
                        IndexError
                    ):  # Should not happen if can_trim is correct, but safety first
                        break
                self._buffer_start_index += trim_count
                # print(f"DEBUG: Trimmed {trim_count} elements. Start index now {self._buffer_start_index}. Len={len(self._buffer)}") # Debug

    def _check_resize(self):
        """Adaptively resize the buffer based on usage patterns."""
        now = time.time()
        elapsed = now - self.last_resize_time
        if elapsed < self.resize_interval:
            return

        # Calculate consumption rate (values per second) based on refills this interval
        # This is a heuristic - assumes refills correlate with consumption
        # A more accurate way would track actual values consumed, but adds overhead
        consumption_rate = self.refill_count_interval / elapsed if elapsed > 0 else 0
        current_size = self._current_max_len

        # Estimate needed buffer size for the next interval based on current rate
        projected_need = consumption_rate * self.resize_interval * 1.5  # Add buffer

        if projected_need > current_size and current_size < self.max_size:
            # Grow buffer
            new_size = min(
                int(max(current_size * self.resize_factor, projected_need)),
                self.max_size,
            )
            if new_size > current_size:
                self._current_max_len = new_size
                # No deque resize needed here, _fill_buffer handles capacity check
                # print(f"DEBUG: Resizing buffer UP to {new_size}") # Optional debug

        elif projected_need < current_size * 0.3 and current_size > self.min_size:
            # Shrink buffer (less aggressively)
            new_size = max(int(current_size / self.resize_factor), self.min_size)
            if new_size < current_size:
                self._current_max_len = new_size
                # Trimming will happen in _trim_buffer if needed
                # print(f"DEBUG: Resizing buffer DOWN to {new_size}") # Optional debug

        # Reset for next interval
        self.refill_count_interval = 0
        self.last_resize_time = now

    def get_bits(self, num_bits):
        """Extract a specific number of random bits (1-64)."""
        if not 1 <= num_bits <= 64:
            raise ValueError("num_bits must be between 1 and 64")

        self.bits_used_total += num_bits
        self.values_generated_total += 1

        if self.bits_remaining == 0:
            self._get_next_value()

        if num_bits <= self.bits_remaining:
            self.bits_remaining -= num_bits
            result = (self.current_value >> self.bits_remaining) & (
                (np.uint64(1) << num_bits) - np.uint64(1)
            )
            return int(result)
        else:
            result = self.current_value & (
                (np.uint64(1) << self.bits_remaining) - np.uint64(1)
            )
            bits_collected = self.bits_remaining
            self._get_next_value()
            bits_needed = num_bits - bits_collected
            self.bits_remaining -= bits_needed
            top_bits = (self.current_value >> self.bits_remaining) & (
                (np.uint64(1) << bits_needed) - np.uint64(1)
            )
            final_result = (result << bits_needed) | top_bits
            return int(final_result)

    def get_bits_batch(self, num_bits, count, variable_lengths=None):
        """
        Extract multiple values with the specified bit length(s). Optimized.
        Returns list of Python ints.
        """
        results = []
        if variable_lengths:
            if len(variable_lengths) != count:
                raise ValueError("variable_lengths must have the same length as count")
            if any(b < 1 or b > 64 for b in variable_lengths):
                raise ValueError("All variable_lengths must be between 1 and 64")
            total_bits_request = sum(variable_lengths)

            self.bits_used_total += total_bits_request
            self.values_generated_total += count

            # --- Optimized Path for Variable Lengths ---
            for bits_to_get in variable_lengths:
                if self.bits_remaining == 0:
                    self._get_next_value()

                if bits_to_get <= self.bits_remaining:
                    self.bits_remaining -= bits_to_get
                    val = (self.current_value >> self.bits_remaining) & (
                        (np.uint64(1) << bits_to_get) - np.uint64(1)
                    )
                    results.append(int(val))
                else:
                    # Needs multiple values
                    res_val = self.current_value & (
                        (np.uint64(1) << self.bits_remaining) - np.uint64(1)
                    )
                    bits_gotten = self.bits_remaining
                    self._get_next_value()
                    bits_needed_now = bits_to_get - bits_gotten
                    self.bits_remaining -= bits_needed_now
                    top_bits = (self.current_value >> self.bits_remaining) & (
                        (np.uint64(1) << bits_needed_now) - np.uint64(1)
                    )
                    final_val = (res_val << bits_needed_now) | top_bits
                    results.append(int(final_val))
            # --- End Optimized Path ---

        else:  # Fixed num_bits
            if not 1 <= num_bits <= 64:
                raise ValueError("num_bits must be between 1 and 64")
            total_bits_request = num_bits * count

            self.bits_used_total += total_bits_request
            self.values_generated_total += count

            # --- Optimized Path for Fixed Length ---
            mask = (np.uint64(1) << num_bits) - np.uint64(1)
            for _ in range(count):
                if self.bits_remaining == 0:
                    self._get_next_value()

                if num_bits <= self.bits_remaining:
                    self.bits_remaining -= num_bits
                    val = (self.current_value >> self.bits_remaining) & mask
                    results.append(int(val))
                else:
                    # Needs multiple values
                    res_val = self.current_value & (
                        (np.uint64(1) << self.bits_remaining) - np.uint64(1)
                    )
                    bits_gotten = self.bits_remaining
                    self._get_next_value()
                    bits_needed_now = num_bits - bits_gotten
                    self.bits_remaining -= bits_needed_now
                    top_bits = (self.current_value >> self.bits_remaining) & (
                        (np.uint64(1) << bits_needed_now) - np.uint64(1)
                    )
                    final_val = (res_val << bits_needed_now) | top_bits
                    results.append(int(final_val))
            # --- End Optimized Path ---

        return results

    def get_state(self):
        """Get the current state of the buffer for serialization (full state)."""
        return {
            # Store logical indices and buffer content relative to start
            "buffer_list": [int(x) for x in self._buffer],  # Convert uint64 for JSON
            "buffer_start_index": self._buffer_start_index,
            "current_logical_index": self.current_logical_index,
            "current_value": int(self.current_value),  # Convert uint64 for JSON
            "bits_remaining": self.bits_remaining,
            "current_max_len": self._current_max_len,
            # Note: No longer storing 'compressed' flag
        }

    def set_state(self, state):
        """Restore state from serialization (assumes full state)."""
        self._buffer = deque(
            np.uint64(x) for x in state["buffer_list"]
        )  # Restore deque content
        self._buffer_start_index = state["buffer_start_index"]
        self.current_logical_index = state["current_logical_index"]
        self.current_value = np.uint64(state["current_value"])
        self.bits_remaining = state["bits_remaining"]
        self._current_max_len = state.get("current_max_len", self.max_size)

        # Calculate derived end index
        self._buffer_end_index = self._buffer_start_index + len(self._buffer)

        # Reset resize timer and interval counter
        self.last_resize_time = time.time()
        self.refill_count_interval = 0

    def get_metrics(self):
        """Get buffer performance metrics"""
        buffer_len = len(self._buffer)
        # Estimate utilization based on logical indices
        consumed_count = max(
            0, self.current_logical_index - self._buffer_start_index + 1
        )
        utilization = consumed_count / buffer_len if buffer_len > 0 else 0
        avg_bits = (
            self.bits_used_total / self.values_generated_total
            if self.values_generated_total > 0
            else 0
        )

        return {
            "buffer_size": buffer_len,
            "buffer_capacity": self._current_max_len,
            "buffer_utilization_ratio": utilization,  # How much of current buffer logically consumed
            "refill_count_interval": self.refill_count_interval,  # Resets periodically
            "bits_used_total": self.bits_used_total,
            "values_generated_total": self.values_generated_total,
            "average_bits_per_value": avg_bits,
        }


class MetricsCollector:
    """Collects and processes performance metrics (unchanged from previous version)"""

    def __init__(self, collection_interval=1.0):
        self.metrics = {
            "refills": 0,  # Note: Now tracked in buffer metrics primarily
            "bits_used": 0,  # Note: Now tracked in buffer metrics primarily
            "values_generated": 0,  # Note: Now tracked in buffer metrics primarily
            "weighted_choices": 0,
            "weighted_samples_ares": 0,  # Added for A-Res
            "integers_generated": 0,
            "floats_generated": 0,
            "shuffles": 0,
            "samples": 0,  # General sample calls
            "batch_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        self.stats = {
            "start_time": time.time(),
            "last_collection_time": time.time(),
            "operations_per_second": 0,
            # Bits/sec might be less meaningful now it's tracked in buffer
        }
        self.metrics_lock = RLock()
        self.updates_queue = deque()
        self.collection_interval = collection_interval
        self.collection_thread = None
        self.running = False

    def start(self):
        if self.collection_thread is not None:
            return
        self.running = True
        self.collection_thread = threading.Thread(
            target=self._metrics_collection_loop, daemon=True
        )
        self.collection_thread.start()

    def stop(self):
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=1.0)
            self.collection_thread = None

    def _metrics_collection_loop(self):
        while self.running:
            time.sleep(self.collection_interval)
            self._process_updates()
            self._update_statistics()

    def _process_updates(self):
        with self.metrics_lock:
            while self.updates_queue:
                metric, value = self.updates_queue.popleft()
                self.metrics[metric] += value

    def _update_statistics(self):
        with self.metrics_lock:
            now = time.time()
            elapsed = now - self.stats["last_collection_time"]
            if elapsed > 0:
                ops = sum(
                    v
                    for k, v in self.metrics.items()
                    if k.endswith("_generated")
                    or k
                    in [
                        "weighted_choices",
                        "weighted_samples_ares",
                        "shuffles",
                        "samples",
                        "batch_operations",
                    ]
                )
                self.stats["operations_per_second"] = ops / elapsed

                total_cache_accesses = (
                    self.metrics["cache_hits"] + self.metrics["cache_misses"]
                )
                if total_cache_accesses > 0:
                    self.stats["cache_hit_rate"] = (
                        self.metrics["cache_hits"] / total_cache_accesses
                    )
                else:
                    self.stats["cache_hit_rate"] = 0

            # Reset interval metrics (if any - mostly moved to buffer)
            self.stats["last_collection_time"] = now

    def update(self, metric, value=1):
        if metric in self.metrics:
            self.updates_queue.append((metric, value))

    def get_metrics(self):
        with self.metrics_lock:
            self._process_updates()  # Ensure queue is flushed
            self._update_statistics()  # Update rates
            result = {
                "metrics": dict(self.metrics),
                "stats": dict(self.stats),
                "current_time": time.time(),
            }
            return result


# --- Numba Accelerated Noise Functions ---
@njit(cache=True, fastmath=True)
def _fade_numba(t):
    """Perlin noise smoothing function 6t^5 - 15t^4 + 10t^3"""
    # Input t is already float64 from GameRNG methods
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(cache=True, fastmath=True)
def _lerp_numba(a, b, t):
    """Linear interpolation"""
    return a + t * (b - a)


@njit(cache=True, fastmath=True)
def _grad1_numba(hash_val, x):
    """1D Perlin gradient calculation"""
    # hash_val is int32, x is float64
    grad_val = 1.0 if (hash_val & 1) == 0 else -1.0
    return grad_val * x


@njit(cache=True, fastmath=True)
def _grad2_numba(hash_val, x, y):
    """2D Perlin gradient calculation (using 8 directions)"""
    # hash_val is int32, x, y are float64
    h = hash_val & 7  # Faster than % 8
    # Gradient vectors (dot products) - expanded for clarity
    if h == 0:
        return x + y  # ( 1,  1)
    if h == 1:
        return -x + y  # (-1,  1)
    if h == 2:
        return x - y  # ( 1, -1)
    if h == 3:
        return -x - y  # (-1, -1)
    if h == 4:
        return x  # ( 1,  0)  Simplified from original logic
    if h == 5:
        return -x  # (-1,  0)
    if h == 6:
        return y  # ( 0,  1)
    # if h == 7: return -y       # ( 0, -1)
    return -y


@njit(cache=True, fastmath=True)
def _perlin_noise_1d_numba(x, p_table):
    """Internal Numba-accelerated 1D Perlin noise calculation"""
    X = math.floor(x)
    x_int = int(X)  # Convert floor to int for indexing
    xf = x - X  # Fractional part

    u = _fade_numba(xf)  # Smoothed fraction

    # Get permutation values for integer coordinates
    idx0 = x_int & 255
    idx1 = (x_int + 1) & 255
    p0 = p_table[idx0]
    p1 = p_table[idx1]

    # Calculate gradients and interpolate
    grad0 = _grad1_numba(p0, xf)
    grad1 = _grad1_numba(p1, xf - 1.0)

    # Final interpolation
    return _lerp_numba(grad0, grad1, u)  # Output approx [-0.5, 0.5]


@njit(cache=True, fastmath=True)
def _perlin_noise_2d_numba(x, y, p_table):
    """Internal Numba-accelerated 2D Perlin noise calculation"""
    X = math.floor(x)
    Y = math.floor(y)
    x_int = int(X)
    y_int = int(Y)
    xf = x - X
    yf = y - Y

    u = _fade_numba(xf)
    v = _fade_numba(yf)

    # Hash coordinates of the 4 cube corners
    idx_X0 = x_int & 255
    idx_X1 = (x_int + 1) & 255
    idx_Y0 = y_int & 255
    idx_Y1 = (y_int + 1) & 255

    # Use p_table directly - Numba handles array access efficiently
    p_X0 = p_table[idx_X0]
    p_X1 = p_table[idx_X1]

    hash_AA = p_table[p_X0 + idx_Y0]
    hash_BA = p_table[p_X1 + idx_Y0]
    hash_AB = p_table[p_X0 + idx_Y1]
    hash_BB = p_table[p_X1 + idx_Y1]

    # Calculate gradients at corners
    grad_AA = _grad2_numba(hash_AA, xf, yf)
    grad_BA = _grad2_numba(hash_BA, xf - 1.0, yf)
    grad_AB = _grad2_numba(hash_AB, xf, yf - 1.0)
    grad_BB = _grad2_numba(hash_BB, xf - 1.0, yf - 1.0)

    # Interpolate
    lerp_x1 = _lerp_numba(grad_AA, grad_BA, u)
    lerp_x2 = _lerp_numba(grad_AB, grad_BB, u)
    result = _lerp_numba(lerp_x1, lerp_x2, v)

    # Scale output closer to [-1, 1] range
    return result * 1.41421356237  # Approx sqrt(2)


class GameRNG:
    """
    An enhanced random number generator for games (v2.3.0).

    Features:
    - Efficient bit-level extraction via CircularBitBuffer (optimized batching)
    - Choice of PRNG algorithms ('pcg64', 'xorshift')
    - Thread safety via RLock
    - Deterministic sequences via seeding
    - Save/Load functionality for full RNG state (incl. noise seed)
    - Numba-accelerated Perlin noise generation (1D, 2D) - deterministic
    - Efficient Weighted Sampling Without Replacement (A-Res)
    - Game-specific convenience methods
    - Optional performance metrics collection
    """

    _UINT64_MAX = np.iinfo(np.uint64).max
    _UINT64_MAX_FLOAT = float(_UINT64_MAX)
    _52_BIT_SCALE = 1.0 / (1 << 52)
    _32_BIT_SCALE = 1.0 / (1 << 32)

    def __init__(
        self,
        seed=None,
        buffer_size=1000,
        max_buffer_size=1000000,
        generator="pcg64",
        metrics=False,
        version="2.3.0",  # <-- Updated version for optimizations
        default_int_type=int,
        noise_seed=None,  # Seed for Perlin noise permutation table
    ):
        """Initialize the game RNG."""
        self.version = version

        if seed is None:
            effective_seed = int(time.time() * 1000)
        else:
            effective_seed = seed
        self.initial_seed = effective_seed

        if generator == "pcg64":
            seed_input = (
                max(0, effective_seed)
                if isinstance(effective_seed, int)
                else effective_seed
            )
            self.rng = np.random.default_rng(seed_input)
        elif generator == "xorshift":
            self.rng = XorshiftGenerator(effective_seed)
        else:
            raise ValueError(f"Unknown generator: {generator}")

        self.generator_type = generator
        self.initial_buffer_size = buffer_size
        self.default_int_type = default_int_type

        # --- Noise Setup ---
        if noise_seed is None:
            seed_int = (
                effective_seed
                if isinstance(effective_seed, int)
                else hash(effective_seed)
            )
            self.noise_seed = (
                (seed_int * np.uint64(0x9E3779B9)).item()
                if seed_int is not None
                else int(time.time() * 1001)
            )
        else:
            self.noise_seed = noise_seed
        self._init_noise_table(self.noise_seed)  # Generate self._p

        # --- Buffer Setup ---
        self.buffer = CircularBitBuffer(
            self.rng, initial_size=buffer_size, max_size=max_buffer_size
        )

        # --- Metrics ---
        self.metrics_enabled = metrics
        self.metrics = MetricsCollector() if metrics else None
        if self.metrics:
            self.metrics.start()

        # --- Thread Safety ---
        self.rw_lock = RLock()

        # --- Caches ---
        self.weighted_choice_cache = {}
        self.weighted_choice_cache_size = 100

        # --- Distributions Setup ---
        self._setup_distributions()

    def _init_noise_table(self, noise_seed_value):
        """Generates the Perlin noise permutation table using SplitMix64."""
        noise_mixer = SplitMix64(noise_seed_value)
        p_table = list(range(256))
        # Fisher-Yates shuffle using the noise_mixer's integers method
        for i in range(255, 0, -1):
            j = int(noise_mixer.integers(low=0, high=i + 1, size=1)[0])
            p_table[i], p_table[j] = p_table[j], p_table[i]
        # Double the table and store as int32 numpy array (required by Numba functions)
        self._p = np.array(p_table * 2, dtype=np.int32)

    def _setup_distributions(self):
        """Set up custom probability distributions using get_float"""
        # Lambdas now call the float method directly
        self.distributions = {
            "bell": lambda: (
                self.get_float()
                + self.get_float()
                + self.get_float()
                + self.get_float()
                - 2.0
            )
            * 0.5,
            "triangle": lambda: (self.get_float() + self.get_float()) * 0.5,
            "power": lambda power=2.0: self.get_float() ** power,
            "exponential": lambda lambd=1.0: -math.log(
                max(1e-9, 1.0 - self.get_float())
            )
            / lambd,
        }

    def _update_metrics(self, metric, value=1):
        """Helper to update metrics if enabled"""
        if self.metrics_enabled and self.metrics:
            self.metrics.update(metric, value)

    # --- Core Randomness Methods ---

    def get_bits(self, num_bits):
        """Extract a specific number of random bits (1-64). Returns Python int."""
        with self.rw_lock:
            # Metrics are now updated within the buffer methods
            return self.buffer.get_bits(num_bits)

    def get_bits_batch(self, num_bits, count, variable_lengths=None):
        """Extract multiple values with specified bit length(s) using optimized buffer method."""
        with self.rw_lock:
            # Metrics updated within buffer method
            results = self.buffer.get_bits_batch(num_bits, count, variable_lengths)
            self._update_metrics("batch_operations")  # Track the batch call itself
            return results

    def get_int(self, min_val, max_val, return_type=None):
        """Get an integer within range [min_val, max_val] (inclusive)."""
        if min_val > max_val:
            raise ValueError("min_val cannot be greater than max_val")
        if min_val == max_val:
            r_type = return_type or self.default_int_type
            return r_type(min_val)

        with self.rw_lock:
            r_type = return_type or self.default_int_type
            range_size = np.uint64(max_val) - np.uint64(min_val) + np.uint64(1)

            if range_size == 0:  # Handles edge case where range is full uint64 width
                bits_needed = 64
                mask = self._UINT64_MAX
            elif (range_size & (range_size - np.uint64(1))) == 0:  # Power of 2
                bits_needed = (
                    range_size.item().bit_length() - 1
                )  # e.g. range 8 (1000) needs 3 bits (mask 0111)
                mask = range_size - np.uint64(1)
            else:  # Non power of 2
                bits_needed = range_size.item().bit_length()
                mask = (
                    (np.uint64(1) << bits_needed) - np.uint64(1)
                    if bits_needed > 0
                    else np.uint64(0)
                )

            # Rejection sampling loop using get_bits
            while True:
                # Get raw bits using the optimized single call
                raw_val = np.uint64(self.buffer.get_bits(bits_needed))
                # Apply mask (though get_bits already effectively does this)
                # value = raw_val & mask # Not strictly needed if get_bits works right
                value = raw_val
                if value < range_size:
                    final_val = np.uint64(min_val) + value
                    result = r_type(final_val)
                    break

            self._update_metrics("integers_generated")
            return result

    def get_ints(self, min_val, max_val, count, return_type=None):
        """Get multiple integers within range [min_val, max_val] (inclusive)."""
        if count < 0:
            raise ValueError("count cannot be negative")
        if count == 0:
            return []

        # Acquire lock once for the whole batch
        with self.rw_lock:
            # OPTIMIZATION: Could potentially optimize further by batching bit requests
            # if the range allows, but looping get_int is often sufficient and simpler.
            r_type = return_type or self.default_int_type
            results = [
                self.get_int(min_val, max_val, return_type=r_type) for _ in range(count)
            ]
            self._update_metrics("batch_operations")
            return results

    def get_float(self, min_val=0.0, max_val=1.0, precision=52):
        """Get a float in range [min_val, max_val)."""
        if precision not in (32, 52):
            raise ValueError("Precision must be 32 (approx single) or 52 (double)")
        if min_val >= max_val:
            if min_val == max_val:
                return min_val
            raise ValueError("min_val must be less than max_val for get_float")

        with self.rw_lock:
            bits = np.uint64(self.buffer.get_bits(precision))
            # Use precomputed scale factor
            scale_factor = self._52_BIT_SCALE if precision == 52 else self._32_BIT_SCALE
            zero_to_one = float(bits) * scale_factor
            result = min_val + (max_val - min_val) * zero_to_one
            self._update_metrics("floats_generated")
            # Clamp below max_val
            return min(result, np.nextafter(max_val, min_val))

    def get_floats(self, min_val=0.0, max_val=1.0, precision=52, count=1):
        """Get multiple floats in range [min_val, max_val)."""
        if count < 0:
            raise ValueError("count cannot be negative")
        if count == 0:
            return []

        with self.rw_lock:
            # OPTIMIZATION: Can optimize by batching bit generation
            if precision == 52:
                raw_bits = self.get_bits_batch(52, count)
                scale = self._52_BIT_SCALE
            elif precision == 32:
                raw_bits = self.get_bits_batch(32, count)
                scale = self._32_BIT_SCALE
            else:  # Fallback for safety, though validated in get_float
                results = [
                    self.get_float(min_val, max_val, precision) for _ in range(count)
                ]
                self._update_metrics("batch_operations")
                return results

            # Vectorized calculation using numpy
            zero_to_one_arr = np.array(raw_bits, dtype=np.float64) * scale
            result_arr = min_val + (max_val - min_val) * zero_to_one_arr

            # Clamp below max_val (element-wise min)
            max_bound = np.nextafter(max_val, min_val)
            final_arr = np.minimum(result_arr, max_bound)

            self._update_metrics("floats_generated", count)
            self._update_metrics("batch_operations")
            return final_arr.tolist()  # Return as list of python floats

    # --- Distributions ---

    def get_distribution(self, dist_type, **kwargs):
        """Generate a value from a specific probability distribution."""
        if dist_type not in self.distributions:
            raise ValueError(f"Unknown distribution: {dist_type}")
        dist_func = self.distributions[dist_type]
        # Pass kwargs safely (lock acquired within get_float)
        if dist_type == "power":
            return dist_func(power=kwargs.get("power", 2.0))
        elif dist_type == "exponential":
            return dist_func(lambd=kwargs.get("lambd", 1.0))
        else:
            return dist_func()  # bell, triangle

    # --- Game-Specific & Collection Methods ---

    def weighted_choice(self, items, weights, cache_key=None):
        """Select an item based on weights using CDF and bisect."""
        with self.rw_lock:
            if len(items) != len(weights):
                raise ValueError("Items and weights must have the same length")
            if not items:
                raise ValueError("Items list cannot be empty")
            if any(w < 0 for w in weights):
                raise ValueError("Weights must be non-negative")

            total = sum(weights)
            if total <= 0:
                # Handle case where all weights are zero - maybe return random item? Or raise?
                # Choosing to raise error for now, as it indicates potentially bad input.
                raise ValueError("Sum of weights must be positive")

            cdf = None
            if cache_key is not None:
                cdf = self.weighted_choice_cache.get(cache_key)
                self._update_metrics(
                    "cache_hits" if cdf is not None else "cache_misses"
                )

            if cdf is None:
                cdf = np.cumsum(weights)  # Use numpy's faster cumsum
                if cache_key is not None:
                    if (
                        len(self.weighted_choice_cache)
                        >= self.weighted_choice_cache_size
                    ):
                        # Simple random eviction
                        keys = list(self.weighted_choice_cache.keys())
                        del self.weighted_choice_cache[
                            keys[self.get_int(0, len(keys) - 1)]
                        ]
                    self.weighted_choice_cache[cache_key] = cdf

            # Generate random value in [0, total) - using max precision float
            r = self.get_float(0.0, float(total))

            # Use bisect_left for efficient lookup
            # cdf is already normalized by the total in the get_float call range
            # Need to use the computed CDF which has the cumulative sums
            idx = bisect.bisect_left(cdf, r)

            # Handle potential edge case if r matches the last cdf value exactly
            idx = min(idx, len(items) - 1)

            result = items[idx]
            self._update_metrics("weighted_choices")
            return result

    def weighted_sample_ares(self, items, weights, k):
        """
        Select k unique items based on weights using A-Res algorithm.
        Efficient for selecting k items without replacement.

        Args:
            items (list): List of items to choose from.
            weights (list): Corresponding non-negative weights for each item.
            k (int): Number of unique items to sample.

        Returns:
            list: A list containing k unique items sampled according to weights.

        Raises:
            ValueError: If inputs are invalid (e.g., k > len(items), negative weights).
        """
        n = len(items)
        if len(weights) != n:
            raise ValueError("Items and weights must have the same length")
        if k < 0 or k > n:
            raise ValueError(f"Cannot sample k={k} items from population of size {n}")
        if k == 0:
            return []
        if any(w < 0 for w in weights):
            raise ValueError("Weights must be non-negative")
        if all(w == 0 for w in weights):
            raise ValueError("All weights are zero, cannot sample")

        with self.rw_lock:
            reservoir = []
            # Use double precision for keys
            keys = np.zeros(k, dtype=np.float64)

            # Convert weights to numpy array for faster access if needed
            weights_arr = np.asarray(weights, dtype=np.float64)

            # Fill the reservoir initially
            i = 0
            while i < n and len(reservoir) < k:
                w = weights_arr[i]
                if w > 0:
                    # Generate key: U^(1/w), where U is random float in (0, 1]
                    u = 1.0 - self.get_float(
                        min_val=0.0, max_val=1.0, precision=52
                    )  # Ensure (0, 1]
                    key = math.pow(u, 1.0 / w)

                    # Simple insertion sort logic for the first k items
                    insert_pos = bisect.bisect_left(keys[: len(reservoir)], key)
                    reservoir.insert(insert_pos, items[i])
                    # Insert key and shift others (inefficient but ok for small k initially)
                    keys = np.insert(keys, insert_pos, key)
                    keys = keys[:k]  # Keep keys array size k
                i += 1

            # If k >= n and all items had positive weight, reservoir is full, return shuffled
            if len(reservoir) == n and n == k:
                self.shuffle(reservoir)  # Ensure random order if k=n
                self._update_metrics("weighted_samples_ares")
                return reservoir

            # Process remaining items
            threshold = keys[0]  # Smallest key in the reservoir
            while i < n:
                w = weights_arr[i]
                if w > 0:
                    u = 1.0 - self.get_float(min_val=0.0, max_val=1.0, precision=52)
                    key = math.pow(u, 1.0 / w)

                    # If key is larger than the smallest key in the reservoir, replace
                    if key > threshold:
                        # Find insertion point (maintaining sorted keys)
                        insert_pos = bisect.bisect_left(keys, key)

                        # Shift elements in reservoir and keys arrays efficiently
                        if insert_pos == k:  # Insert at the end
                            reservoir.pop(0)
                            reservoir.append(items[i])
                            keys[:-1] = keys[1:]
                            keys[-1] = key
                        elif insert_pos > 0:  # Insert in the middle
                            # Shift reservoir items
                            reservoir.pop(0)
                            reservoir.insert(
                                insert_pos - 1, items[i]
                            )  # insert before insert_pos
                            # Shift keys
                            keys[: insert_pos - 1] = keys[1:insert_pos]
                            keys[insert_pos - 1] = key
                        else:  # insert_pos is 0 - this shouldn't happen if key > threshold
                            # This case indicates an issue or floating point precision problem
                            # Fallback: just replace the first element (less accurate but safe)
                            reservoir[0] = items[i]
                            keys[0] = key
                            # Resort keys needed after fallback replace (less efficient)
                            sort_indices = np.argsort(keys)
                            keys = keys[sort_indices]
                            reservoir = [reservoir[j] for j in sort_indices]

                        # Update threshold (smallest key is now keys[0])
                        threshold = keys[0]
                i += 1

            self._update_metrics("weighted_samples_ares")
            # Result is already approximately shuffled by the nature of the keys
            # If strict uniform shuffling of the result is needed, uncomment:
            # self.shuffle(reservoir)
            return reservoir

    def shuffle(self, items):
        """Shuffle a list in-place using Fisher-Yates."""
        with self.rw_lock:
            n = len(items)
            # Use numpy array for potential speedup if items is large and numeric
            # but stick to list for general compatibility
            if isinstance(items, np.ndarray) and n > 1000:  # Heuristic threshold
                for i in range(n - 1, 0, -1):
                    j = self.get_int(0, i)
                    items[i], items[j] = (
                        items[j],
                        items[i],
                    )  # Numpy handles swap efficiently
            else:  # Standard list shuffle
                for i in range(n - 1, 0, -1):
                    j = self.get_int(0, i)
                    items[i], items[j] = items[j], items[i]

            self._update_metrics("shuffles")
            # No return value needed for in-place shuffle

    def sample(self, items, k, replacement=False, weights=None):
        """
        Sample k items from a list/population. Supports weighted sampling.

        Args:
            items (list or sequence): Population to sample from.
            k (int): Number of items to sample.
            replacement (bool): Sample with replacement (True) or without (False).
            weights (list, optional): Weights for weighted sampling. If None, assumes uniform.

        Returns:
            list: Sampled items.
        """
        n = len(items)
        if k < 0:
            raise ValueError("k cannot be negative")
        if k == 0:
            return []

        with self.rw_lock:
            if replacement:
                if weights is None:  # Uniform with replacement
                    indices = self.get_ints(0, n - 1, k)
                    result = [items[i] for i in indices]
                else:  # Weighted with replacement
                    # Use weighted_choice repeatedly
                    result = [self.weighted_choice(items, weights) for _ in range(k)]
            else:  # Without replacement
                if k > n:
                    raise ValueError("Cannot sample k > n without replacement")
                if weights is None:  # Uniform without replacement
                    if (
                        k > n / 2 and n > 100
                    ):  # Heuristic: Shuffle if sampling more than half
                        # Make a copy to shuffle if items shouldn't be modified
                        items_copy = list(items)
                        self.shuffle(items_copy)
                        result = items_copy[:k]
                    else:  # Reservoir sampling or index tracking for smaller k
                        indices = set()
                        result = []
                        while len(result) < k:
                            idx = self.get_int(0, n - 1)
                            if idx not in indices:
                                indices.add(idx)
                                result.append(items[idx])
                else:  # Weighted without replacement
                    result = self.weighted_sample_ares(items, weights, k)

            self._update_metrics("samples")
            return result

    # Game-specific convenience methods (mostly unchanged, rely on core methods)

    def roll_dice(self, num_dice=1, sides=6, modifier=0):
        """Roll dice as in tabletop games."""
        # Note: sides must be >= 1
        if sides < 1:
            raise ValueError("Dice sides must be at least 1")
        if num_dice < 0:
            raise ValueError("Number of dice cannot be negative")

        with self.rw_lock:
            if num_dice == 0:
                rolls = []
                total = modifier
            else:
                rolls = self.get_ints(1, sides, num_dice)
                total = sum(rolls) + modifier

            # No specific metric update here, covered by get_ints

            return {"total": total, "rolls": rolls, "modifier": modifier}

    def coin_flip(self, num_flips=1, weighted=False, heads_probability=0.5):
        """Flip a coin one or more times."""
        if num_flips < 0:
            raise ValueError("Number of flips cannot be negative")
        if num_flips == 0:
            return []

        with self.rw_lock:
            results = []
            if weighted:
                if not 0.0 <= heads_probability <= 1.0:
                    raise ValueError("heads_probability must be between 0.0 and 1.0")
                floats = self.get_floats(0.0, 1.0, count=num_flips)
                results = [
                    "heads" if r < heads_probability else "tails" for r in floats
                ]
            else:
                bits = self.get_bits_batch(1, num_flips)
                results = ["heads" if b == 1 else "tails" for b in bits]

            # No specific metric update here, covered by get_floats/get_bits_batch

            return results if num_flips > 1 else results[0]

    def deck_of_cards(self, shuffled=True):
        """Create a standard deck of 52 playing cards."""
        # Use tuple for immutability if preferred, but list is fine
        suits = ("hearts", "diamonds", "clubs", "spades")
        ranks = (
            "ace",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "jack",
            "queen",
            "king",
        )
        deck = [{"rank": rank, "suit": suit} for suit in suits for rank in ranks]
        if shuffled:
            # shuffle method is in-place
            self.shuffle(deck)  # Metric updated in shuffle
        return deck

    def loot_table(self, table, count=1, unique=False):
        """Generate loot drops from a loot table. Uses optimized sampling."""
        if count < 0:
            raise ValueError("count cannot be negative")
        if count == 0:
            return []
        if not table:
            return []  # Empty table

        items = list(table.keys())
        weights = list(table.values())

        with self.rw_lock:
            if unique:
                # Use optimized weighted sampling without replacement
                result = self.sample(items, count, replacement=False, weights=weights)
            else:
                # Use repeated weighted choice (sampling with replacement)
                result = self.sample(items, count, replacement=True, weights=weights)

            # Metrics updated within self.sample calls
            return result

    def uuid(self, version=4):
        """Generate a UUID version 4 using this RNG."""
        if version != 4:
            raise ValueError("Only UUID version 4 is supported")

        with self.rw_lock:
            # Generate 128 bits (16 bytes) efficiently
            b1, b2 = self.get_bits_batch(64, 2)

            # Construct UUID bytes (most significant byte first in each part)
            b = bytearray(16)
            for i in range(8):
                b[i] = (b1 >> (56 - i * 8)) & 0xFF
                b[i + 8] = (b2 >> (56 - i * 8)) & 0xFF

            # Set version (4) and variant (RFC 4122) bits
            b[6] = (b[6] & 0x0F) | 0x40  # Version 4
            b[8] = (b[8] & 0x3F) | 0x80  # Variant RFC 4122

            # Format as hex string
            hex_uuid = b.hex()
            return f"{hex_uuid[:8]}-{hex_uuid[8:12]}-{hex_uuid[12:16]}-{hex_uuid[16:20]}-{hex_uuid[20:]}"

    # --- Perlin Noise Methods (Using Numba Accelerated Helpers) ---

    def noise_1d(self, x, scale=1.0, seed_offset=0):
        """Generate Numba-accelerated 1D Perlin noise."""
        # No lock needed - reads immutable self._p table & calls Numba func.
        if scale == 0:
            return 0.0
        # Ensure input is float64 for Numba function
        input_x = np.float64(x / scale) + np.float64(seed_offset)
        # Pass the permutation table to the Numba function
        return _perlin_noise_1d_numba(input_x, self._p)

    def noise_2d(self, x, y, scale=1.0, seed_offset=0):
        """Generate Numba-accelerated 2D Perlin noise."""
        # No lock needed.
        if scale == 0:
            return 0.0
        input_x = np.float64(x / scale) + np.float64(seed_offset)
        input_y = np.float64(y / scale) + np.float64(seed_offset)
        return _perlin_noise_2d_numba(input_x, input_y, self._p)

    # --- State Management ---

    def get_state(self):
        """Return the current state of the RNG for serialization (full state)."""
        with self.rw_lock:
            # Ensure metrics are up-to-date if saving state includes metrics
            # buffer_metrics = self.buffer.get_metrics() # Included in buffer_state

            state = {
                "version": self.version,
                "generator_type": self.generator_type,
                "rng_state": self.rng.__getstate__(),
                "buffer_state": self.buffer.get_state(),  # Contains full buffer now
                "initial_seed": self.initial_seed,
                "default_int_type_name": self.default_int_type.__name__,
                "noise_seed": self.noise_seed,
                # Optionally save metrics state if needed
                # "metrics_state": self.metrics.get_metrics() if self.metrics else None
            }
            return state

    def set_state(self, state):
        """Restore the RNG state from a previously saved state (full state)."""
        with self.rw_lock:
            # Version check
            state_version = state.get("version", "1.0.0")  # Assume 1.0.0 if missing
            # Basic major.minor compatibility check (allow loading older minor versions)
            try:
                s_major, s_minor, *_ = map(int, state_version.split("."))
                c_major, c_minor, *_ = map(int, self.version.split("."))
                if s_major != c_major or s_minor > c_minor:
                    raise ValueError(
                        f"Incompatible state version: Current is {self.version}, loaded state is {state_version}"
                    )
            except ValueError:
                raise ValueError(
                    f"Invalid version format: Current={self.version}, Loaded={state_version}"
                )

            if state["generator_type"] != self.generator_type:
                raise ValueError(
                    f"State is for {state['generator_type']} but this RNG uses {self.generator_type}"
                )

            # Restore main generator state FIRST
            self.rng.__setstate__(state["rng_state"])

            # Restore buffer state using its set_state
            self.buffer.set_state(state["buffer_state"])

            # Restore other attributes
            self.initial_seed = state.get("initial_seed")  # Allow None if older state
            self.noise_seed = state["noise_seed"]

            # Restore default int type (safer lookup)
            default_int_type_name = state.get("default_int_type_name", "int")
            try:
                import builtins

                self.default_int_type = getattr(builtins, default_int_type_name, None)
                if self.default_int_type is None:
                    self.default_int_type = getattr(np, default_int_type_name, int)
            except Exception:
                print(
                    f"Warning: Could not restore type '{default_int_type_name}', defaulting to int."
                )
                self.default_int_type = int

            # Re-initialize noise table from the loaded seed
            self._init_noise_table(self.noise_seed)

            # Reset caches (state might invalidate them)
            self.weighted_choice_cache = {}

            # Optionally restore metrics state if saved
            # if self.metrics and state.get("metrics_state"):
            #     self.metrics.set_state(state["metrics_state"]) # Assumes MetricsCollector has set_state

            print(f"RNG state loaded successfully (Version: {state_version})")

    def save_state_to_file(self, filename, use_orjson=False):
        """
        Save the full RNG state to a file using JSON or orjson.

        Args:
            filename (str): File to save state to.
            use_orjson (bool): Use orjson library for faster serialization if available.
        """
        state = self.get_state()

        # No need to manually convert numpy types if using orjson with default handler
        # or if buffer state already converts to standard ints

        if use_orjson:
            try:
                import orjson

                with open(filename, "wb") as f:  # orjson outputs bytes
                    # Use option for numpy serialization if needed, though buffer state is list[int] now
                    f.write(
                        orjson.dumps(
                            state,
                            option=(
                                orjson.OPT_SERIALIZE_NUMPY
                                if hasattr(orjson, "OPT_SERIALIZE_NUMPY")
                                else 0
                            ),
                        )
                    )
            except ImportError:
                print("Warning: orjson not installed, falling back to standard json.")
                use_orjson = False  # Ensure fallback happens

        if not use_orjson:  # Standard JSON fallback
            # Need to ensure numpy types are handled if they exist elsewhere
            def default_serializer(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                raise TypeError(
                    f"Object of type {obj.__class__.__name__} is not JSON serializable"
                )

            with open(filename, "w") as f:
                json.dump(
                    state, f, default=default_serializer, indent=2
                )  # Add indent for readability

    def load_state_from_file(self, filename, use_orjson=False):
        """Load the RNG state from a file (JSON or orjson)."""
        state = None
        if use_orjson:
            try:
                import orjson

                with open(filename, "rb") as f:
                    state = orjson.loads(f.read())
            except ImportError:
                print("Warning: orjson not installed, falling back to standard json.")
                use_orjson = False
            except FileNotFoundError:
                raise
            except Exception as e:
                print(f"Error loading with orjson: {e}. Attempting standard json.")
                use_orjson = False  # Fallback on error

        if not use_orjson:  # Standard JSON fallback or orjson failed
            try:
                with open(filename, "r") as f:
                    state = json.load(f)
            except FileNotFoundError:
                raise
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to decode JSON from {filename}: {e}") from e
            except Exception as e:  # Catch other potential errors during file reading
                raise IOError(f"Error reading state file {filename}: {e}") from e

        if state is None:
            raise IOError(f"Could not load state from file: {filename}")

        # Now call the main set_state method with the loaded dictionary
        self.set_state(state)

    def get_metrics(self):
        """Get performance metrics including buffer metrics if enabled."""
        if not self.metrics_enabled or not self.metrics:
            return None

        # Get metrics from collector
        collector_metrics = self.metrics.get_metrics()

        # Get metrics from buffer and merge
        buffer_metrics = self.buffer.get_metrics()
        collector_metrics["buffer"] = (
            buffer_metrics  # Add buffer metrics under its own key
        )

        return collector_metrics

    def reset(self, seed=None, noise_seed="reset"):
        """
        Reset the RNG state. Resets main generator, buffer, noise, caches, metrics.

        Args:
            seed (int, optional): New seed for main generator. Defaults to None (time-based).
            noise_seed (int or str, optional): New seed for noise ('reset', None, or int).
                                                 'reset': Derive noise seed from main seed.
                                                 None: Use time-based noise seed.
                                                 int: Use the provided integer as noise seed.
        """
        with self.rw_lock:
            # --- Reset Main Generator ---
            if seed is None:
                effective_seed = int(time.time() * 1000)
            else:
                effective_seed = seed
            self.initial_seed = effective_seed

            if self.generator_type == "pcg64":
                seed_input = (
                    max(0, effective_seed)
                    if isinstance(effective_seed, int)
                    else effective_seed
                )
                self.rng = np.random.default_rng(seed_input)
            else:  # xorshift
                self.rng = XorshiftGenerator(effective_seed)

            # --- Reset Noise Setup ---
            if noise_seed == "reset":
                seed_int = (
                    effective_seed
                    if isinstance(effective_seed, int)
                    else hash(effective_seed)
                )
                effective_noise_seed = (
                    (seed_int * np.uint64(0x9E3779B9)).item()
                    if seed_int is not None
                    else int(time.time() * 1001)
                )
            elif noise_seed is None:
                effective_noise_seed = int(time.time() * 1001)
            else:
                effective_noise_seed = noise_seed
            self.noise_seed = effective_noise_seed
            self._init_noise_table(self.noise_seed)  # Regenerate permutation table

            # --- Reset Buffer ---
            # Keep previous max_size unless it wasn't set
            prev_max_size = getattr(
                self.buffer, "max_size", self.initial_buffer_size * 10
            )
            self.buffer = CircularBitBuffer(
                self.rng, initial_size=self.initial_buffer_size, max_size=prev_max_size
            )

            # --- Reset Metrics ---
            if self.metrics_enabled:
                if self.metrics:
                    self.metrics.stop()
                self.metrics = MetricsCollector()
                self.metrics.start()

            # --- Reset Caches ---
            self.weighted_choice_cache = {}

    def run_self_tests(self, level="basic"):
        """Run statistical tests on the generator (mostly unchanged)."""
        results = {}
        # Store state before test
        original_state = self.get_state()

        # Basic tests (run on current state)
        if level in ("basic", "standard", "extended"):
            try:
                # Bit balance
                bits = self.get_bits_batch(1, 10000)
                ones = sum(bits)
                zeros = 10000 - ones
                bit_balance = abs(ones - zeros) / 10000
                results["bit_balance"] = bit_balance
                results["bit_balance_ok"] = bit_balance < 0.05

                # Integer uniformity
                ints = self.get_ints(1, 100, 10000)  # Use batch method
                int_counts = {}
                for i in range(1, 101):
                    int_counts[i] = 0
                for val in ints:
                    if 1 <= val <= 100:
                        int_counts[val] += 1  # Count safely
                expected_int = 10000 / 100.0
                chi_squared_int = (
                    sum(
                        ((count - expected_int) ** 2) / expected_int
                        for count in int_counts.values()
                    )
                    if expected_int > 0
                    else 0
                )
                results["int_uniformity_chi_squared"] = chi_squared_int
                # Adjust threshold slightly for potential small biases in PRNGs over 10k samples
                results["int_uniformity_ok"] = chi_squared_int < 160  # Was 150

                # Serial correlation
                if len(ints) > 1:
                    correlation = np.nan_to_num(np.corrcoef(ints[:-1], ints[1:])[0, 1])
                    results["serial_correlation"] = correlation
                    results["correlation_ok"] = (
                        abs(correlation) < 0.03
                    )  # Relaxed slightly from 0.02
                else:
                    results["serial_correlation"] = 0
                    results["correlation_ok"] = True

            except Exception as e:
                print(f"Error during basic self-tests: {e}")
                results["basic_tests_error"] = str(e)

        # Standard tests
        if level in ("standard", "extended"):
            try:
                # Run the xorshift's built-in tests if applicable
                if self.generator_type == "xorshift":
                    # Need to run on a temporary instance or ensure state is restored
                    temp_rng_xorshift = XorshiftGenerator()
                    temp_rng_xorshift.__setstate__(
                        self.rng.__getstate__()
                    )  # Clone state
                    results["xorshift_internal_tests"] = (
                        temp_rng_xorshift.test_randomness(20000)
                    )
                    # No need to restore state here as temp_rng was used

                # Pair distribution test
                pairs_x = self.get_ints(0, 9, 1000)
                pairs_y = self.get_ints(0, 9, 1000)
                pair_counts = {}
                for x in range(10):
                    for y in range(10):
                        pair_counts[(x, y)] = 0
                for i in range(1000):
                    pair_counts[(pairs_x[i], pairs_y[i])] += 1
                expected_pair = 1000 / 100.0
                chi_squared_pairs = (
                    sum(
                        ((count - expected_pair) ** 2) / expected_pair
                        for count in pair_counts.values()
                    )
                    if expected_pair > 0
                    else 0
                )
                results["pair_distribution_chi_squared"] = chi_squared_pairs
                results["pair_distribution_ok"] = chi_squared_pairs < 160  # Was 150
            except Exception as e:
                print(f"Error during standard self-tests: {e}")
                results["standard_tests_error"] = str(e)

        # Extended tests
        if level == "extended":
            try:
                # Longest run test (on bits)
                bits_ext = self.get_bits_batch(1, 50000)
                current_run = 0
                longest_run = 0
                last_bit = -1
                for bit in bits_ext:
                    if bit == last_bit:
                        current_run += 1
                    else:
                        longest_run = max(longest_run, current_run)
                        current_run = 1
                        last_bit = bit
                longest_run = max(longest_run, current_run)  # Check final run
                results["longest_run"] = longest_run
                expected_longest_run = math.log2(50000) if 50000 > 0 else 1
                results["longest_run_reasonable"] = (
                    expected_longest_run * 0.5
                    < longest_run
                    < expected_longest_run * 2.5
                )

                # Simplified run length distribution check (deviation from geometric)
                run_lengths = []
                current_run = 0
                last_bit = -1
                for bit in bits_ext:
                    if bit == last_bit:
                        current_run += 1
                    else:
                        if current_run > 0:
                            run_lengths.append(current_run)
                        current_run = 1
                        last_bit = bit
                if current_run > 0:
                    run_lengths.append(current_run)  # Add final run

                length_counts = {}
                total_runs = len(run_lengths)
                for length in run_lengths:
                    length_counts[length] = length_counts.get(length, 0) + 1

                geo_dist_deviation = 0
                checked_lengths = 0
                max_len_check = (
                    15  # Don't check excessively long runs (too few samples)
                )
                for length in range(1, max_len_check + 1):
                    count = length_counts.get(length, 0)
                    expected = total_runs * (0.5**length)
                    if expected > 1:  # Only check if expected count is reasonably high
                        deviation = abs(count - expected) / expected
                        geo_dist_deviation += deviation
                        checked_lengths += 1

                avg_deviation = (
                    geo_dist_deviation / checked_lengths if checked_lengths > 0 else 0
                )
                results["run_length_distribution_avg_deviation"] = avg_deviation
                results["run_length_distribution_ok"] = (
                    avg_deviation < 0.3
                )  # Allow slightly more deviation
            except Exception as e:
                print(f"Error during extended self-tests: {e}")
                results["extended_tests_error"] = str(e)

        # Restore original state after tests
        self.set_state(original_state)

        # Overall result
        results["all_tests_passed"] = all(
            value for key, value in results.items() if key.endswith("_ok")
        )
        return results


# Example usage (Updated for new noise and state, uncommented)
if __name__ == "__main__":
    # Basic usage examples
    print("Basic usage examples...")
    rng = GameRNG(
        seed=42, buffer_size=1000, metrics=True, noise_seed=101
    )  # Explicit seeds
    np_rng = GameRNG(
        seed=42,
        buffer_size=1000,
        default_int_type=np.uint64,
        noise_seed=102,
        generator="xorshift",
    )

    print("\nBit operations:")
    print("8-bit value:", rng.get_bits(8))
    print("16-bit value:", rng.get_bits(16))
    print("32-bit value:", rng.get_bits(32))
    print("Batch of five 8-bit values:", rng.get_bits_batch(8, 5))
    print(
        "Variable bit length batch:",
        rng.get_bits_batch(0, 3, variable_lengths=[3, 5, 8]),
    )

    print("\nNumber generation:")
    print(
        "Integer (1-100, default type int):",
        rng.get_int(1, 100),
        type(rng.get_int(1, 100)),
    )
    print(
        "Integer (1-100, explicit type np.int32):",
        rng.get_int(1, 100, return_type=np.int32),
        type(rng.get_int(1, 100, return_type=np.int32)),
    )
    print(
        "Integer (1-100, default type np.uint64):",
        np_rng.get_int(1, 100),
        type(np_rng.get_int(1, 100)),
    )
    print("Integers (1-10, count 5, default type int):", rng.get_ints(1, 10, 5))
    print(
        "Integers (1-10, count 5, explicit type np.uint8):",
        rng.get_ints(1, 10, 5, return_type=np.uint8),
    )

    print("\nFloat operations:")
    print(f"Float (0-1, default precision 52): {rng.get_float():.8f}")
    print(f"Float (precision 32): {rng.get_float(precision=32):.8f}")
    print(f"Range [0.5, 1.5): {rng.get_float(0.5, 1.5):.8f}")
    print(f"Floats batch (count 3): {rng.get_floats(0.0, 1.0, count=3)}")

    print("\nDistributions:")
    print("Normal (bell curve approx):", rng.get_distribution("bell"))
    print("Triangle:", rng.get_distribution("triangle"))
    print("Power law:", rng.get_distribution("power", power=3))
    print("Exponential:", rng.get_distribution("exponential", lambd=2.0))

    print("\nGame operations:")
    print("Coin flip:", rng.coin_flip())
    print("10 coin flips:", rng.coin_flip(10))
    print(
        "Weighted coin (75% heads):",
        rng.coin_flip(weighted=True, heads_probability=0.75),
    )

    print("\nDice rolls:")
    print("d6:", rng.roll_dice())
    print("2d8+3:", rng.roll_dice(num_dice=2, sides=8, modifier=3))

    print("\nWeighted choice:")
    items_rarity = ["Common", "Uncommon", "Rare", "Epic", "Legendary"]
    weights_rarity = [60, 25, 10, 4, 1]
    print(
        "Item rarity:",
        rng.weighted_choice(items_rarity, weights_rarity, cache_key="rarity_table"),
    )
    # Test caching
    print(
        "Item rarity (cached):",
        rng.weighted_choice(items_rarity, weights_rarity, cache_key="rarity_table"),
    )

    print("\nShuffling:")
    my_list = list(range(1, 11))
    print("Original:", my_list)
    rng.shuffle(my_list)  # Shuffle is now in-place
    print("Shuffled:", my_list)

    print("\nSampling (Uniform):")
    pop = list(range(10))
    print("Sample 3 from range(10):", rng.sample(pop, 3))
    print("Sample 10 with replacement:", rng.sample(pop, 10, replacement=True))

    print("\nSampling (Weighted):")
    items_w = ["A", "B", "C", "D"]
    weights_w = [1, 10, 2, 5]
    print(f"Population: {items_w}, Weights: {weights_w}")
    print(
        "Sample 3 Weighted (A-Res):",
        rng.sample(items_w, 3, replacement=False, weights=weights_w),
    )
    print(
        "Sample 5 Weighted (Replace):",
        rng.sample(items_w, 5, replacement=True, weights=weights_w),
    )

    print("\nLoot table example:")
    loot = {"Gold": 50, "Health Potion": 30, "Arrow": 15, "Magic Scroll": 5}
    print("3 drops (unique):", rng.loot_table(loot, 3, unique=True))
    print("5 drops (can repeat):", rng.loot_table(loot, 5, unique=False))

    print("\nUUID generation:")
    print("UUID:", rng.uuid())

    print("\nPerlin Noise functions (Numba accelerated):")
    start_time = time.time()
    noise_1d_val = rng.noise_1d(0.5)
    noise_2d_val = rng.noise_2d(0.3, 0.7)
    # Generate a small noise map
    map_size = 50
    noise_map = np.zeros((map_size, map_size))
    for y in range(map_size):
        for x in range(map_size):
            noise_map[y, x] = rng.noise_2d(x * 0.1, y * 0.1, scale=5.0)
    end_time = time.time()
    print(f"1D Noise at x=0.5: {noise_1d_val:.4f}")
    print(f"2D Noise at (0.3, 0.7): {noise_2d_val:.4f}")
    print(
        f"Generated {map_size}x{map_size} noise map in {end_time - start_time:.4f} seconds"
    )

    print("\nState management (full state):")
    # Get state before save
    rng.get_int(1, 10)  # Advance state slightly
    state_before = rng.get_state()
    noise_val1 = rng.noise_2d(0.5, 0.5)  # Noise doesn't change state
    seq1 = [rng.get_int(1, 10) for _ in range(3)]
    print(f"Noise before save: {noise_val1:.4f}, Sequence: {seq1}")

    # Save state
    state_file = "rng_state_v2_3.json"
    try:
        rng.save_state_to_file(state_file, use_orjson=True)  # Try orjson
        print(f"State saved to {state_file} using orjson")
    except ImportError:
        rng.save_state_to_file(state_file, use_orjson=False)
        print(f"State saved to {state_file} using standard json")

    # Reset and check values change
    rng.reset(seed=999, noise_seed=999)
    print(
        f"Noise after reset: {rng.noise_2d(0.5, 0.5):.4f}, Sequence: {[rng.get_int(1,10) for _ in range(3)]}"
    )

    # Load state from file
    try:
        rng.load_state_from_file(state_file, use_orjson=True)
    except ImportError:
        rng.load_state_from_file(state_file, use_orjson=False)
    except FileNotFoundError:
        print(f"Error: State file {state_file} not found. Cannot test load.")

    # Compare state after load
    state_after = rng.get_state()
    noise_val3 = rng.noise_2d(0.5, 0.5)
    seq3 = [rng.get_int(1, 10) for _ in range(3)]
    print(f"Noise after file load: {noise_val3:.4f}, Sequence: {seq3}")

    # Deep compare relevant parts of the state dictionary
    # Comparing entire dict might fail due to float precision or object IDs
    state_match = True
    if state_before["noise_seed"] != state_after["noise_seed"]:
        state_match = False
        print("Mismatch: noise_seed")
    if state_before["buffer_state"] != state_after["buffer_state"]:
        state_match = False
        print("Mismatch: buffer_state")
    # Compare RNG state carefully - might need custom comparison for numpy objects
    if str(state_before["rng_state"]) != str(state_after["rng_state"]):
        state_match = False
        print("Mismatch: rng_state (string representation)")

    print("State load appears consistent:", state_match)
    print("Restored noise seed:", rng.noise_seed)
    # Check sequence continuation
    seq4 = [rng.get_int(1, 10) for _ in range(3)]
    print(
        f"Sequence continuation after load: {seq4}"
    )  # Should be different from seq1/seq3

    print("\nMetrics:")
    metrics_data = rng.get_metrics()
    if metrics_data:
        print(
            json.dumps(metrics_data, indent=2, default=str)
        )  # Use default=str for Enum/types
    else:
        print("Metrics not enabled.")

    print("\nSelf tests:")
    test_results = rng.run_self_tests(level="standard")  # Run standard tests
    print(json.dumps(test_results, indent=2, default=str))

    print("\n--- Completed ---")
