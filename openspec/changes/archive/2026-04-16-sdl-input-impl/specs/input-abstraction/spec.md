## ADDED Requirements

### Requirement: nextFrame timing convention
Callers SHALL invoke `window->getInputState()->nextFrame()` after consuming input state for the current frame (typically at frame end). This clears per-frame accumulators (mouse delta, wheel delta) while preserving held-key states.

#### Scenario: nextFrame called at frame end
- **WHEN** a frame loop calls `nextFrame()` after reading mouse delta
- **THEN** the next frame's `getMouseDelta()` SHALL start from `{0, 0}`
