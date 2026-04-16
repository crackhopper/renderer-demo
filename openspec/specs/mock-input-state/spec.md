## ADDED Requirements

### Requirement: MockInputState class
`src/core/input/mock_input_state.hpp` SHALL define a class `MockInputState` in namespace `LX_core` that implements `IInputState`. It SHALL be header-only.

The class SHALL provide writable setters:
- `void setKeyDown(KeyCode k, bool down)`
- `void setMouseButtonDown(MouseButton b, bool down)`
- `void setMousePosition(Vec2f p)`
- `void setMouseDelta(Vec2f d)`
- `void setMouseWheelDelta(float w)`

And implement all `IInputState` pure virtual methods returning the stored values.

Internal storage SHALL use:
- `std::array<bool, static_cast<size_t>(KeyCode::Count)>` for keys (default all false)
- `std::array<bool, static_cast<size_t>(MouseButton::Count)>` for buttons (default all false)
- `Vec2f` for position and delta (default zero)
- `float` for wheel delta (default zero)

#### Scenario: Default state is all-zero
- **WHEN** constructing a `MockInputState` with no modifications
- **THEN** all getters SHALL return false or zero, identical to `DummyInputState`

#### Scenario: Set and get key state
- **WHEN** calling `setKeyDown(KeyCode::W, true)`
- **THEN** `isKeyDown(KeyCode::W)` SHALL return `true` and `isKeyDown(KeyCode::A)` SHALL return `false`

#### Scenario: Set and get mouse button state
- **WHEN** calling `setMouseButtonDown(MouseButton::Left, true)`
- **THEN** `isMouseButtonDown(MouseButton::Left)` SHALL return `true`

#### Scenario: Set and get mouse delta
- **WHEN** calling `setMouseDelta({10.0f, -5.0f})`
- **THEN** `getMouseDelta()` SHALL return `{10.0f, -5.0f}`

#### Scenario: Set and get wheel delta
- **WHEN** calling `setMouseWheelDelta(2.0f)`
- **THEN** `getMouseWheelDelta()` SHALL return `2.0f`

### Requirement: MockInputState nextFrame clears per-frame accumulators
Calling `nextFrame()` SHALL reset mouse delta to `{0, 0}` and wheel delta to `0.0f`. Key states, mouse button states, and mouse position SHALL be preserved.

#### Scenario: nextFrame clears delta but keeps buttons
- **WHEN** mouse delta is `{10, 5}`, left button is down, and `nextFrame()` is called
- **THEN** `getMouseDelta()` SHALL return `{0, 0}` and `isMouseButtonDown(MouseButton::Left)` SHALL remain `true`

#### Scenario: nextFrame clears wheel delta
- **WHEN** wheel delta is `1.0f` and `nextFrame()` is called
- **THEN** `getMouseWheelDelta()` SHALL return `0.0f`

### Requirement: MockInputState does not depend on SDL
`MockInputState` SHALL NOT include any SDL headers or link against SDL libraries. It SHALL only depend on `core/input/input_state.hpp` and standard library headers.

#### Scenario: MockInputState compiles without SDL
- **WHEN** building a translation unit that includes only `mock_input_state.hpp`
- **THEN** compilation SHALL succeed without SDL being available
