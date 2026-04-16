## 1. Sdl3InputState 实现

- [x] 1.1 创建 `src/infra/window/sdl3_input_state.hpp`（类声明、handleSdlEvent、内部状态）
- [x] 1.2 创建 `src/infra/window/sdl3_input_state.cpp`（scancode 映射表 + IInputState 方法实现 + handleSdlEvent 事件处理）
- [x] 1.3 在 `src/infra/CMakeLists.txt` 加入 `sdl3_input_state.cpp`

## 2. SDL Window 接入

- [x] 2.1 修改 `sdl_window.cpp`：Impl 持有 `shared_ptr<Sdl3InputState>`，构造时创建
- [x] 2.2 修改 `Impl::shouldClose()` 在 poll 循环中调用 `inputState->handleSdlEvent(event)`
- [x] 2.3 修改 `Window::getInputState()` 返回真实 Sdl3InputState（移除 DummyInputState）

## 3. 集成测试

- [x] 3.1 创建 `src/test/integration/test_sdl_input.cpp`（手工构造 SDL_Event 测试 key down/up、mouse button、position、delta 累加、wheel、nextFrame 清零）
- [x] 3.2 在 `src/test/CMakeLists.txt` 注册 `test_sdl_input`
- [x] 3.3 构建并运行测试验证通过

## 4. 收尾

- [x] 4.1 更新 REQ-013 实施状态
