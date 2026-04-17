## ADDED Requirements

### Requirement: IInputState reports UI capture flags

`IInputState`（`src/core/input/input_state.hpp`）SHALL 追加两个默认虚方法：

```cpp
virtual bool isUiCapturingMouse() const { return false; }
virtual bool isUiCapturingKeyboard() const { return false; }
```

默认实现 SHALL 返回 `false`，以保证既有调用点与既有实现无需修改即可编译。后续若上层 UI（如 ImGui）声明希望独占鼠标/键盘，具体实现（例如 `Sdl3InputState`）MAY 覆写这两个方法返回 `ImGui::GetIO().WantCaptureMouse` 与 `WantCaptureKeyboard`；本 REQ 不强制 `Sdl3InputState` 在当前版本就接通 ImGui，只规定接口契约。

相机控制器（REQ-015 / REQ-016）与 demo viewer（REQ-019）SHALL 通过这两个方法查询 UI capture 状态；本 REQ 不要求在此处定义具体的消费策略。

#### Scenario: 默认实现返回 false

- **WHEN** 调用任何直接继承 `IInputState` 但未覆写 capture 方法的实现（例如当前 `DummyInputState`）
- **THEN** `isUiCapturingMouse()` 与 `isUiCapturingKeyboard()` SHALL 返回 `false`

#### Scenario: 既有 Dummy/Mock 实现继续通过编译

- **WHEN** 重新编译依赖 `core/input/input_state.hpp` 的所有既有代码（包括 `DummyInputState`、`Sdl3InputState`、`MockInputState`、各相机控制器测试）
- **THEN** 编译 SHALL 成功，且无任何实现被强制覆写这两个方法
