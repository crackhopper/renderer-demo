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

### Requirement: MockInputState supports UI capture override

`MockInputState`（`src/core/input/mock_input_state.hpp`）SHALL 暴露两个额外的 setter：

- `void setUiCapturingMouse(bool capturing)`
- `void setUiCapturingKeyboard(bool capturing)`

并覆写 `IInputState::isUiCapturingMouse()` / `isUiCapturingKeyboard()` 返回各自存储的布尔值。默认值 SHALL 为 `false`，与 `IInputState` 默认行为一致。

此扩展的存在价值：相机控制器（REQ-015 / REQ-016 / REQ-019）在单元/集成测试中需要模拟"UI 正在吃输入"的场景以验证控制器不抢鼠标；`MockInputState` 是这些测试注入输入的唯一通道。

#### Scenario: 默认 UI capture 为 false

- **WHEN** 构造 `MockInputState` 后立即查询
- **THEN** `isUiCapturingMouse()` 与 `isUiCapturingKeyboard()` SHALL 都返回 `false`

#### Scenario: setUiCapturingMouse 影响查询结果

- **WHEN** 调用 `setUiCapturingMouse(true)` 后查询
- **THEN** `isUiCapturingMouse()` SHALL 返回 `true`，`isUiCapturingKeyboard()` SHALL 仍返回 `false`

#### Scenario: setUiCapturingKeyboard 影响查询结果

- **WHEN** 调用 `setUiCapturingKeyboard(true)` 后查询
- **THEN** `isUiCapturingKeyboard()` SHALL 返回 `true`，`isUiCapturingMouse()` SHALL 仍返回 `false`
