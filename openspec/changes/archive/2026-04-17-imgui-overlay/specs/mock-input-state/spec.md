## ADDED Requirements

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
