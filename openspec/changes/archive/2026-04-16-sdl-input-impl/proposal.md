## Why

REQ-012 已定义 `IInputState` 和 `Window::getInputState()`，但 SDL3 后端仍返回 `DummyInputState`，事件循环只处理 `SDL_EVENT_QUIT`。REQ-015/016 的相机控制器和 REQ-017 的 ImGui 输入协调需要真实键鼠状态。本变更将 SDL3 事件接入 `IInputState` 背后，使 `getInputState()` 返回随事件更新的真实输入。

## What Changes

- 新建 `Sdl3InputState`（`src/infra/window/sdl3_input_state.hpp/.cpp`），实现 `IInputState`
- `handleSdlEvent()` 消费 SDL 键盘、鼠标移动、按键、滚轮、quit 事件
- 内部 scancode → `KeyCode` 映射表覆盖 REQ-012 所有枚举值
- `sdl_window.cpp` 的 `Impl` 持有 `Sdl3InputState`，`shouldClose()` 复用同一轮 poll 并更新输入
- `getInputState()` 从返回 dummy 改为返回真实 SDL 输入状态
- GLFW 保持 `DummyInputState` 占位
- 新增集成测试：手工构造 SDL_Event 验证 `Sdl3InputState` 行为

## Capabilities

### New Capabilities
- `sdl-input-state`: SDL3 真实输入状态实现（Sdl3InputState + scancode 映射 + 事件消费）

### Modified Capabilities
- `window-system`: SDL window 的 `getInputState()` 从 dummy 切换为真实实现；`shouldClose()` 复用事件轮询
- `input-abstraction`: 明确 `nextFrame()` 调用时序约定

## Impact

- **代码**：`src/infra/window/` 新增 2 文件，修改 `sdl_window.cpp`
- **构建**：`src/infra/CMakeLists.txt` 加入新 .cpp
- **测试**：新增 `test_sdl_input`
- **GLFW**：不变，继续 dummy
