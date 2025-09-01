# simple_rl/AI - Community NPC AI System

## Purpose

This component implements the core artificial intelligence for **non-adventurer NPCs** residing in persistent, evolving communities within the game world. It focuses on simulating complex social behaviors, daily routines, learning, and adaptation based on individual traits and experiences, distinct from the combat/survival-oriented GOAP AI used for adventurers and monsters (`simple_rl/auto`).

Future plans include potentially allowing the player character to be managed by this AI system when in a "community manager" mode, and enabling NPCs transitioning between adventuring and community life to switch AI systems.

## Core Functionality (Based on v9.py - v8.py is obsolete)

* **Trait System (`TraitProfile`):** Models individual NPC characteristics (Endurance, Ingenuity, Perception, Will, Resonance) that influence behavior, learning, costs, and recovery.
* **Needs Management:** Tracks and influences behavior based on Health, Energy, Thirst, Hunger, and potentially specific Nutrients.
* **Physiological Simulation (`FatigueSystem`, `IllnessSystem`):** Models fatigue accumulation/recovery and the onset/progression/resolution of illnesses, impacted by traits and actions.
* **Habit Learning:**
    * Agents learn multi-step behaviors (**Habits**) by observing recurring sequences of atomic actions (**Behaviors**) in their experience log (`behavior_memory`).
    * Habits are associated with **trigger conditions** (state features or functions).
    * Habits have a **score** reflecting their perceived effectiveness, updated based on contextual outcome evaluation (`_contextual_value_of_outcome`) and decayed over time if unused.
    * Low-scoring or long-unused habits are pruned.
    * Traits influence habit formation (Ingenuity) and reinforcement (Resonance).
* **Planning & Decision Making (`decide_day_plan`):**
    * Agents plan their day by iteratively selecting triggered, affordable habits based on calculated **utility**.
    * Utility considers predicted impact on needs, resource gain/loss, costs (time, energy), and trait modifiers (`_calculate_habit_utility`).
    * **Adaptive Impact Estimation:** Habit impact (`_estimate_habit_impact`) uses both pre-defined behavior impacts and dynamically learned average costs/gains derived from the agent's historical task performance (`task_stats`), modulated by traits like Ingenuity.
    * **Identity & Dissonance (`SelfConcept`):** Agent identity influences habit utility. Executing actions misaligned with identity can cause dissonance, incurring penalties (e.g., energy cost), mitigated by the Will trait.
* **Experience Memory (`ExperienceMemory`):** Records outcomes of actions linked to preceding states, enabling prediction and potentially faster learning (influenced by Ingenuity).

## Key Files

* **`v9.py`**: Contains the current implementation of the `AgentF` class, `Habit` system, `TraitProfile`, and associated subsystems (`FatigueSystem`, `IllnessSystem`, `ExperienceMemory`, `SelfConcept`). (Note: `v8.py` is considered obsolete).

## Dependencies

* **Python:** 3.x
* **Core Libraries:** `numpy`
* **Project Dependencies:**
    * `rng_utils.game_rng.GameRNG`: For all random number generation.
    * Likely depends on data structures/constants defined elsewhere (e.g., `Home`, `Field`, `CROPS`, `Weather`, `Calendar`, `Behavior` definitions - potentially shared with other modules or defined centrally). *Exact dependencies need verification during integration.*

## Status & Integration

This AI system is under active development. It is intended to drive NPCs within community environments and interact with game systems like resource management, time, weather, and potentially social structures (not yet implemented). It is distinct from the `simple_rl/auto` GOAP AI. Normalization between the trait systems used here and the player/adventurer trait system is a planned future task.
