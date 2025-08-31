# Rendering Options

The renderer supports additional controls for how remembered tiles fade over time.

- **memory_fade_variance** – A float in the range `[0.0, 1.0]` that applies per-tile
  random offsets in hue and desaturation when fading tiles in memory.  The
  randomness is derived from the game state's RNG and tile coordinates so the
  pattern remains stable frame-to-frame but differs for each map.  Higher values
  produce greater colour variation between tiles.
- **memory_noise_level** – A float in the range `[0.0, 1.0]` controlling the
  chance that a remembered tile uses an alternate "noisy" glyph set.  At `0.0`
  standard glyphs are used; at `1.0` all faded tiles swap to their noisy
  counterparts based on intensity. Noise decisions are based on tile positions
  and the game's RNG seed, keeping them consistent between frames while varying
  across maps.

These options are available on the `RenderConfig` dataclass and can be supplied
by systems constructing the renderer configuration.
