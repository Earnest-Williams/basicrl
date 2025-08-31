# Prioritized AI worklist (estimates in developer-days)

## 1) Unit tests expansion (2-4 days)
   - Add tests for Habit learning, ExperienceMemory, SelfConcept dissonance, and trait influence.
   - Integrate tests into CI.

## 2) Action schema adoption & validation (1-2 days)
   - Add validators in adapters and fail-fast behavior for malformed actions.

## 3) Deterministic action resolver improvements (2-4 days)
   - Handle swaps, pushing, blocking rules, and integrate with the entity registry and GameMap.
   - Add more tests for complex conflict scenarios.

## 4) Single-threaded action application + merge policy (3-5 days)
   - Implement authoritative merging step that consumes actions from parallel workers and applies them deterministically.
   - Ensure atomicity of interactions (combat, triggers).

## 5) Ray parallel AI loop integration (4-7 days)
   - Implement worker functions, chunking strategy, and robust merging of AI state and actions.
   - Add monitoring and graceful failure handling.

## 6) Profiling and performance optimization (2-3 days)
   - Profile AI decision making overhead and optimize hotpaths.
   - Tune chunk sizes and parallel worker count.

## 7) CI integration and automated testing (1-2 days)
   - Ensure all AI tests run in CI pipeline.
   - Add performance regression tests.