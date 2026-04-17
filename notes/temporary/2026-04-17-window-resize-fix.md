# 窗口 resize 后不再渲染 — 排查与修复

> 2026-04-17，`demo_scene_viewer` 首次人工验收过程中出现的问题。记录
> 一下排查路径和最终修复，方便以后遇到"某个能力加完了但窗口行为怪怪的"
> 时能直接抓关键点。

## 现象

在 Windows 下跑 `demo_scene_viewer`，窗口能启动、能看见模型、相机交互
正常。但只要调整窗口大小，画面就卡住不再更新，而进程仍然活着（UI 输入
事件仍被吃掉）。最小化再还原也一样。

## 最先想到但不是根因的几个点

- **没调 `updateSwapchainImageCount`**：担心 swapchain 重建后
  ImGui backend 仍用旧 image count。结果排查完发现 `rebuildSwapchain()`
  已经在调用，不是这里的问题。
- **`vkResetFences` 作用在未 signal 的 fence 上**：看着可疑，实际上
  `vkResetFences` 对"已经是 unsignaled"的 fence 是 no-op，既不出错也不死锁。
- **视口 Y-flip**：有个 `LX_RENDER_FLIP_VIEWPORT_Y` debug 开关，以及
  projection matrix 的 `r.m[1][1] = +f`（不是 `-f`）。看起来 fishy，但
  demo 初启动明显能看见画面，说明这些在稳态下是对的，不影响 resize。

## 真正的根因

`LX_infra::Window::getWidth()` / `getHeight()` 返回的是**构造时**记录的
像素尺寸，并且**永远不更新**：

```cpp
// src/infra/window/sdl_window.cpp（修复前）
struct Window::Impl {
  int width;
  int height;
  ...
  Impl(const char *t, int w, int h) : width(w), height(h), ... {}
};
int Window::getWidth() const { return pImpl->width; }
int Window::getHeight() const { return pImpl->height; }
```

从 resize 到下一帧崩盘的完整链路是：

1. 用户拖动窗口边缘，系统通知 SDL 窗口尺寸变了。
2. 下一次 `draw()` 调用 `vkAcquireNextImageKHR()` 返回
   `VK_ERROR_OUT_OF_DATE_KHR` —— 这是正常的，说明 swapchain 需要重建。
3. `rebuildSwapchain()` 调 `swapchain->rebuild(...)`。
4. `VulkanSwapchain::rebuild(...)` 内部用
   `VkExtent2D{ m_window->getWidth(), m_window->getHeight() }` 构建新的
   swapchain —— **但这两个值仍是原始窗口尺寸，不是现在的真实尺寸**。
5. 新 swapchain 大小和实际 surface 不匹配，下一次 `vkAcquireNextImageKHR()`
   继续返回 `VK_ERROR_OUT_OF_DATE_KHR`。
6. 无限循环。驱动层面的尺寸永远对不上，画面就卡死了。

## 为什么最小化也会坏

窗口最小化瞬间，`getWidth()/getHeight()` 也会返回 0。即使我们修好了上面
的"stale cached size"问题，`vkCreateSwapchainKHR` 在 0×0 extent 上行为
是 UB / driver 特定的，我们应该主动跳过那一帧。

## 修复

### 1. 让 `Window::getWidth() / getHeight()` 做真实查询

把 "获取尺寸" 从"读 Impl 缓存字段"改成"每次问 SDL"：

```cpp
// src/infra/window/sdl_window.cpp（修复后）
int Window::getWidth() const {
  int w = pImpl->width;
  int h = pImpl->height;
  SDL_GetWindowSizeInPixels(pImpl->window, &w, &h);
  pImpl->width = w;
  pImpl->height = h;
  return w;
}
int Window::getHeight() const {
  int w = pImpl->width;
  int h = pImpl->height;
  SDL_GetWindowSizeInPixels(pImpl->window, &w, &h);
  pImpl->width = w;
  pImpl->height = h;
  return h;
}
```

`SDL_GetWindowSizeInPixels` 是 local syscall，每帧一两次无所谓。GLFW 路径
用 `glfwGetFramebufferSize(pImpl->window, ...)` 做同样的事情。

选这条路而不是"监听 SDL resize 事件更新 pImpl->width"的原因：

- 少一个状态同步点（事件没到 / 事件顺序被打乱都不会影响正确性）。
- 不必在 `shouldClose()` 的 poll 循环里加 resize 事件分发 —— 那条路径
  已经同时跑 `ImGui_ImplSDL3_ProcessEvent` 和 `Sdl3InputState::handleSdlEvent`，
  再加 resize 处理会让它变拥挤。
- 作为"多后端同构"的接口行为，"调用时返回当前值"是更强的契约。

### 2. `draw()` / `rebuildSwapchain()` 守住零尺寸

```cpp
// src/backend/vulkan/vulkan_renderer.cpp（修复后）
void draw() override {
  if (m_window && (m_window->getWidth() <= 0 || m_window->getHeight() <= 0)) {
    return; // 最小化或拖动过程中，跳过这一帧
  }
  ...
}

void rebuildSwapchain() {
  if (m_window && (m_window->getWidth() <= 0 || m_window->getHeight() <= 0)) {
    return; // 零尺寸下 swapchain 重建会失败，等下一帧再试
  }
  swapchain->waitIdle();
  swapchain->rebuild(resourceManager->getRenderPass());
  m_gui.updateSwapchainImageCount(swapchain->getImageCount());
}
```

配套把 `WindowPtr m_window` 存到 `VulkanRendererImpl` 里，`initialize()`
时保存一下。

## 验证路径

- Linux headless 下 `cmake --build build --target demo_scene_viewer` 通过，
  依赖链路编译无回归。
- Windows 下人工验收：窗口放大 / 缩小 / 最小化还原 / 拖动过程 —— 画面都
  能持续跟随重建，不再卡死。

## 下一次再遇到类似症状时的检查清单

1. **先验证 swapchain 拿到的 extent 是当前窗口真实尺寸**。`getExtent()`
   值对 vs 错是头号原因。在 `rebuild()` 里加一行 `std::cerr` 把 rawExtent
   打出来立刻能看到差异。
2. **`VK_ERROR_OUT_OF_DATE_KHR` 处理路径里是否早返回但没真正 rebuild**。
   有些实现把 "reset fence + return" 和 "调用 rebuild" 错开了位置，
   结果一个 resize 后的错误状态被滑到下一帧重新返回错误，看起来像
   "永远卡着"。
3. **零尺寸防御**。最小化是最容易暴露的。
4. **ImGui 的 `updateSwapchainImageCount`** —— 如果新 swapchain 的 image
   count 和旧的不同（部分驱动会这样），没通知 ImGui backend 会在下一次
   `RenderDrawData` 里访问越界。这一条在本次 REQ-017 落地时就加好了，
   但迁移到其他后端时别忘了复制过去。

## 相关代码

- `src/infra/window/sdl_window.cpp` — SDL live-query getWidth/getHeight
- `src/infra/window/glfw_window.cpp` — GLFW 对应实现
- `src/backend/vulkan/vulkan_renderer.cpp` — 零尺寸守卫 + `m_window` 成员
- `src/backend/vulkan/details/render_objects/swapchain.cpp` — `rebuild()`
  使用 `m_window->getWidth()/getHeight()`
- `src/infra/gui/imgui_gui.cpp` — `updateSwapchainImageCount(imageCount)`
