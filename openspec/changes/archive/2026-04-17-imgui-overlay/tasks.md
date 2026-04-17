## 1. Gui 接线补齐

- [x] 1.1 修改 `src/infra/gui/gui.hpp`：扩 `InitParams` 加 `nativeWindowHandle`、`renderPass`、`swapchainImageCount`；把 `endFrame()` 改为 `endFrame(VkCommandBuffer cmd)`
- [x] 1.2 修改 `src/infra/gui/imgui_gui.cpp`：`init()` 按 CheckVersion / CreateContext / StyleColorsDark / ImplSDL3_InitForVulkan / 创建 descriptor pool / ImplVulkan_Init 顺序接线；`Impl` 持有 `VkDescriptorPool` 与 `VkDevice`
- [x] 1.3 在 `imgui_gui.cpp` 补 `shutdown()` 对称释放：ImplVulkan_Shutdown / ImplSDL3_Shutdown / 销毁 descriptor pool / DestroyContext
- [x] 1.4 在 `imgui_gui.cpp` 实现 `beginFrame()`（SDL3 NewFrame + Vulkan NewFrame + ImGui::NewFrame）与 `endFrame(cmd)`（Render + RenderDrawData(drawData, cmd, VK_NULL_HANDLE)），对空 drawData 或 `TotalVtxCount == 0` 早退

## 2. Window native handle

- [x] 2.1 修改 `src/core/platform/window.hpp`：追加 `virtual void* getNativeHandle() const = 0`
- [x] 2.2 修改 `src/infra/window/window.hpp` 声明新 override；`src/infra/window/sdl_window.cpp` 返回内部 `SDL_Window*`；`src/infra/window/glfw_window.cpp` 返回内部 `GLFWwindow*`
- [x] 2.3 验证 `core/platform/window.hpp` 不 include SDL/GLFW 头（仅 `void*` 出现在接口）

## 3. SDL 事件把 ImGui 接入 poll 循环

- [x] 3.1 修改 `src/infra/window/sdl_window.cpp` 的 `shouldClose()`：在 poll 循环内、`handleSdlEvent` 之前调用 `ImGui_ImplSDL3_ProcessEvent(&event)`
- [x] 3.2 保证 `ImGui_ImplSDL3_ProcessEvent` 在 `Gui::init()` 尚未完成时安全：以 `ImGui::GetCurrentContext() != nullptr` 作为 ready 真值源（Gui::init 成功后 CreateContext 已调，符合条件；未 init 时跳过 forward）
- [x] 3.3 保证 `shouldClose()` 仍只 poll 一次，close / input / ImGui 共享同一轮事件

## 4. Input abstraction 新增 UI capture

- [x] 4.1 修改 `src/core/input/input_state.hpp`：在 `IInputState` 中追加 `virtual bool isUiCapturingMouse() const { return false; }` 与 `virtual bool isUiCapturingKeyboard() const { return false; }`
- [x] 4.2 验证 `DummyInputState` 无需改动即可继承默认行为
- [x] 4.3 修改 `src/core/input/mock_input_state.hpp`：增加 `setUiCapturingMouse(bool)` / `setUiCapturingKeyboard(bool)` 与对应存储；覆写两个 capture 方法返回存储值
- [x] 4.4 保留 `Sdl3InputState` 当前实现不动；本 REQ 不强制把 `ImGui::GetIO().WantCapture*` 接通（可作为后续 REQ 的工作）

## 5. VulkanRenderer 驱动 overlay

- [x] 5.1 修改 `src/backend/vulkan/vulkan_renderer.hpp`：`VulkanRenderer` 暴露 `void setDrawUiCallback(std::function<void()>)`（下穿到 PImpl）
- [x] 5.2 修改 `src/backend/vulkan/vulkan_renderer.cpp` 的内部 impl：新增 `infra::Gui m_gui` 与 `std::function<void()> m_drawUiCallback`
- [x] 5.3 在 impl 的 `initialize()` 末尾，装配 `Gui::InitParams`（`nativeWindowHandle = window->getNativeHandle()`, `renderPass = m_swapchainRenderPass`, `swapchainImageCount = m_swapchainImages.size()`，其他字段用现有 VulkanDevice 值）后调用 `m_gui.init(params)`
- [x] 5.4 在 impl 的 `shutdown()` 最前面（device 释放之前）调用 `m_gui.shutdown()`
- [x] 5.5 在 impl 的 `draw()` 命令录制中按 "begin cmd → begin render pass → `m_gui.beginFrame()` → UI 回调 → 场景 draw → `m_gui.endFrame(cmd)` → end render pass → end cmd" 顺序改造
- [x] 5.6 如果 `m_drawUiCallback` 为空则跳过调用；不要求回调非空

## 6. 集成测试

- [x] 6.1 新建 `src/test/integration/test_imgui_overlay.cpp`，添加三个用例：`gui_init_succeeds_after_renderer_initialize`、`draw_with_ui_callback_does_not_crash`、`ui_capture_flags_default_to_false_without_imgui`
- [x] 6.2 在 `src/test/CMakeLists.txt` 注册 `test_imgui_overlay`（参照现有 integration test 注册方式）
- [x] 6.3 本地跑一次 `cmake --build build --target test_imgui_overlay && ./build/src/test/test_imgui_overlay` 验证通过（CI 无 display 情况下 `LX_SKIP_GRAPHICS_TESTS=1` 或 `$DISPLAY=""` 自动跳过 renderer 依赖用例，UI capture 默认行为断言通过）

## 7. 收尾

- [x] 7.1 更新 `docs/requirements/017-imgui-overlay.md` 的 `## 实施状态` 段落
- [x] 7.2 在 `notes/vulkan-backend/05-command-recording-and-draw.md` 补 overlay-in-swapchain 顺序与 `setDrawUiCallback` 入口的说明
- [x] 7.3 `cmake --build build` 全量通过；input / SDL input / camera controller / engine loop / imgui overlay 相关测试均 PASS（无 display 环境下 renderer 用例自动跳过）
