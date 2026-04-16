## Why

当前窗口系统没有向 `core/` 层暴露任何统一的输入状态接口。`Window` 只有窗口尺寸和 graphics handle，SDL 事件循环只消费 `SDL_EVENT_QUIT`，其余输入事件全部被丢弃。REQ-015/016 的相机控制器、REQ-017 的 ImGui 输入协调都需要一个稳定的 `core` 层输入入口。本变更只做最小前置版本：立住只读输入快照接口和枚举，让后续 SDL3 实现和相机控制器有共同依赖。

## What Changes

- 新建 `src/core/input/` 目录，包含 `KeyCode` 枚举、`MouseButton` 枚举、`IInputState` 纯虚接口
- 新建 `DummyInputState`（全零实现），用于占位和 headless 测试
- `Window` 接口新增 `getInputState()` 纯虚方法
- `LX_infra::Window` 同步实现 `getInputState()` override，返回 `DummyInputState`
- SDL/GLFW window 实现持有共享 `DummyInputState`，不在本变更中读取真实事件
- 新增集成测试验证 `DummyInputState` 零值语义

## Capabilities

### New Capabilities
- `input-abstraction`: `KeyCode`/`MouseButton` 枚举、`IInputState` 接口、`DummyInputState` 实现

### Modified Capabilities
- `window-system`: `Window` 接口新增 `getInputState()` 纯虚方法

## Impact

- **代码**：`src/core/input/`（4 个新头文件）、`src/core/platform/window.hpp`（新增纯虚方法）、`src/infra/window/window.hpp` 和 `sdl_window.cpp`/`glfw_window.cpp`（override 实现）
- **构建**：`src/core/CMakeLists.txt` 需纳入新头文件
- **测试**：新增 `test_input_state` 集成测试
- **依赖**：依赖 `core/math/vec.hpp`（`Vec2f`），无外部新增依赖
