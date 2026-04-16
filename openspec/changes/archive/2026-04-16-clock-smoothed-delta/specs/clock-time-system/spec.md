## ADDED Requirements

### Requirement: Clock existing contract
`Clock` SHALL provide the following methods with specified semantics:
- `void tick()` — advance the clock
- `float deltaTime() const` — time since previous tick (seconds)
- `double totalTime() const` — elapsed time since first tick (seconds)
- `uint64_t frameCount() const` — number of completed frames

#### Scenario: First tick returns zero delta
- **WHEN** `tick()` is called for the first time
- **THEN** `deltaTime()` SHALL return `0.0f`, `totalTime()` SHALL return `0.0`, and `frameCount()` SHALL return `0`

#### Scenario: Second tick returns nonzero delta
- **WHEN** `tick()` is called a second time after a measurable delay
- **THEN** `deltaTime()` SHALL be greater than `0.0f` and `frameCount()` SHALL be `1`

#### Scenario: totalTime monotonically increases
- **WHEN** `tick()` is called multiple times
- **THEN** `totalTime()` SHALL be greater than or equal to its previous value after each tick

### Requirement: Smoothed delta time
`Clock` SHALL provide `float smoothedDeltaTime() const` that returns a sliding-window average of recent `deltaTime()` values over a fixed window of 60 frames.

#### Scenario: Smoothed delta falls back to deltaTime when no samples
- **WHEN** `smoothedDeltaTime()` is called before any samples have been recorded (after first tick only)
- **THEN** the result SHALL equal `deltaTime()`

#### Scenario: Smoothed delta averages recent samples
- **WHEN** multiple ticks have been performed with varying delays
- **THEN** `smoothedDeltaTime()` SHALL return the arithmetic mean of the most recent min(sampleCount, 60) delta values

#### Scenario: Smoothed window uses partial data when under 60 samples
- **WHEN** fewer than 60 ticks have occurred (beyond the first)
- **THEN** `smoothedDeltaTime()` SHALL average only the samples that exist, not divide by 60

### Requirement: tick writes to smoothing window
`tick()` SHALL write the current frame's `deltaTime()` into the smoothing ring buffer starting from the second tick. The first tick SHALL NOT write to the buffer.

#### Scenario: First tick does not write to buffer
- **WHEN** `tick()` is called for the first time
- **THEN** the smoothing buffer SHALL remain empty

#### Scenario: Subsequent ticks populate buffer
- **WHEN** `tick()` is called for the Nth time (N > 1)
- **THEN** the smoothing buffer SHALL contain N-1 samples

### Requirement: Clock integration test
`src/test/integration/test_clock.cpp` SHALL verify first-tick zero delta, second-tick nonzero delta, totalTime monotonicity, smoothedDeltaTime fallback, and smoothedDeltaTime averaging.

#### Scenario: All clock tests pass
- **WHEN** running `test_clock`
- **THEN** all assertions SHALL pass
