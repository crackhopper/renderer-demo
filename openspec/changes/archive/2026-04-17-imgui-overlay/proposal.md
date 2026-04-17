## Why

REQ-017 指出当前 ImGui 集成停留在半成品：`Gui::init()` 缺少 `CreateContext` / SDL backend init / descriptor pool / render pass，`Gui::endFrame()` 用 `VK_NULL_HANDLE` 调 `RenderDrawData` 无法工作，`VulkanRenderer` 未持有 `Gui`，`Window` 缺 `getNativeHandle()`，SDL 事件也未 forward 给 ImGui。相机控制器与调试面板都等待一个可用的 overlay 闭环。本变更以 swapchain render pass 内 overlay 的最小形态打通 ImGui，并补齐 UI capture 协调点。

## What Changes

- `Gui::InitParams` 扩展为 `{instance, physicalDevice, device, graphicsQueueFamilyIndex, presentQueueFamilyIndex, graphicsQueue, presentQueue, surface, nativeWindowHandle, renderPass, swapchainImageCount}`
- `Gui::init()` 按 `CreateContext → StyleColorsDark → ImGui_ImplSDL3_InitForVulkan → 创建 descriptor pool → ImGui_ImplVulkan_Init` 顺序接线
- `Gui::shutdown()` 对称释放 descriptor pool 并销毁 context
- **BREAKING**：`Gui::endFrame()` 签名改为 `endFrame(VkCommandBuffer cmd)`，内部调 `ImGui::Render()` + `ImGui_ImplVulkan_RenderDrawData(drawData, cmd, VK_NULL_HANDLE)`；drawData 为空时跳过提交
- `VulkanRenderer` 持有 `Gui` 成员，`initialize()` 结束时构造并 init；`draw()` 录制顺序：`gui->beginFrame()` → UI 回调 → 场景绘制 → `gui->endFrame(cmd)` → `endRenderPass`
- 新增 `VulkanRenderer::setDrawUiCallback(std::function<void()>)`（仅 `VulkanRenderer`，不下沉到 `Renderer` 基类）
- `Window` 新增 `virtual void* getNativeHandle() const = 0`，SDL 返回 `SDL_Window*`，GLFW 返回 `GLFWwindow*`；`core` 层保持不依赖 SDL/GLFW 头
- SDL 事件循环在 `handleSdlEvent` 之前调用 `ImGui_ImplSDL3_ProcessEvent(&event)`，close/input/ImGui 三条消费路径共用同一次 poll
- `IInputState` 追加默认虚方法 `isUiCapturingMouse()` / `isUiCapturingKeyboard()` 默认返回 `false`；`DummyInputState` 继承默认，`MockInputState` 允许覆写；`Sdl3InputState` 允许写入 `ImGui::GetIO().WantCapture*` 作为真值源
- 新增 `src/test/integration/test_imgui_overlay.cpp` 覆盖 init、draw-loop、ui-capture 默认行为三组场景

## Capabilities

### New Capabilities

- `imgui-overlay`: 定义 `VulkanRenderer` 中 ImGui overlay 的接线顺序、UI 回调入口、overlay 位于 swapchain render pass 尾部的约定

### Modified Capabilities

- `gui-system`: `InitParams` 扩字段、`init/shutdown` 完整接线、`endFrame(cmd)` 签名变更、descriptor pool 所有权归 `Gui`
- `window-system`: `Window` 新增 `getNativeHandle()` 纯虚方法，SDL/GLFW 各自实现；SDL 事件循环增加 `ImGui_ImplSDL3_ProcessEvent` forward
- `input-abstraction`: 追加 `isUiCapturingMouse` / `isUiCapturingKeyboard` 默认虚方法
- `mock-input-state`: `MockInputState` 需允许覆写 UI capture 标志供相机控制器测试使用
- `renderer-backend-vulkan`: `VulkanRenderer` 持有 `Gui` 并在 draw loop 中驱动 overlay，新增 `setDrawUiCallback`

## Impact

- **代码**：`src/infra/gui/` 2 文件重写 init/endFrame；`src/backend/vulkan/vulkan_renderer.{hpp,cpp}` 新增 `Gui` 与 UI 回调；`src/core/platform/window.hpp` 与 `src/infra/window/{sdl_window,glfw_window}.cpp` 新增 native handle 返回；`src/infra/window/sdl3_input_state.cpp` 在事件路径上 forward 给 ImGui；`src/core/input/input_state.hpp` 与 `mock_input_state.hpp` 追加 UI capture
- **构建**：无新增模块；ImGui backend 源文件已由 external imgui 构建逻辑编译，不新增 `src/infra/gui/CMakeLists.txt`
- **测试**：新增 `test_imgui_overlay`，注册到 `src/test/CMakeLists.txt`
- **依赖**：依赖 `REQ-012`（输入抽象）、`REQ-013`（SDL 真实输入）；下游 `REQ-018`（调试面板）、`REQ-019`（demo UI 协调）
- **非目标**：不做独立 FrameGraph overlay pass、不做 docking/viewports、不做自定义字体、不修改 `Renderer` 抽象基类
