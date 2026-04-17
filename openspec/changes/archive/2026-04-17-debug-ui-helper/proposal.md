## Why

REQ-017 已经把 ImGui overlay 接进 `VulkanRenderer`，demo 可以通过 `setDrawUiCallback()` 直接写原生 ImGui。但每个 demo 都要自己从 `ImGui::Begin` / `SliderFloat` / `ColorEdit3` 起手，会重复做 `Vec3f` / `Vec4f` / `StringID` 的桥接，重复写 FPS / 相机 / 方向光 panel，命名和默认布局也会逐步发散。本变更在 `infra/gui/` 下加一组薄 helper 把常用 widget 桥好，并把 FPS / Camera / DirectionalLight 三个 panel 组合好，保持调用方随时可以混用原生 ImGui。

## What Changes

- 新增 `src/infra/gui/debug_ui.hpp` 与 `src/infra/gui/debug_ui.cpp`，命名空间 `LX_infra::debug_ui`
- 基础桥接：`dragVec3` / `dragVec4`（基于 `Vec3f` / `Vec4f`），`sliderFloat` / `sliderInt`，`colorEdit3` / `colorEdit4`，`labelText` / `labelFloat` / `labelInt`，`labelStringId`（通过 `GlobalStringTable::getName` 解析）
- Panel 容器：`beginPanel/endPanel` 统一默认位置/大小策略，`beginSection/endSection` 统一折叠入口，`separatorText`
- 常用 panel：`renderStatsPanel(Clock&)`、`cameraPanel(const char*, Camera&)`、`directionalLightPanel(const char*, DirectionalLight&)`
- `cameraPanel` 只编辑字段本身，**不隐式调用** `updateMatrices()`；是否刷新由调用方决定
- `directionalLightPanel` 基于 `light.ubo->param.{dir,color}` 真实数据布局工作，修改后 helper 调用 `light.ubo->setDirty()`
- 不提供 `materialPanel()` / scene graph inspector / docking / 表格 DSL / 响应式 panel 描述器
- 把 `gui/debug_ui.cpp` 加入 `src/infra/CMakeLists.txt` 的 `INFRA_SOURCES`
- 新增 `src/test/integration/test_debug_ui_smoke.cpp`（链接级验证 + 可创建 ImGui context 时的最小 smoke case），注册到 `src/test/CMakeLists.txt`

## Capabilities

### New Capabilities

- `debug-ui-helper`: `LX_infra::debug_ui` 薄 helper 层——基础桥接、panel 容器、内置 `renderStatsPanel` / `cameraPanel` / `directionalLightPanel` 三个组合面板；helper 保持无状态、可组合、可与原生 ImGui 混用；不进入 `core` 层

### Modified Capabilities

- 无（只加一个新 helper 模块，不改动既有 capability 的 spec 行为）

## Impact

- **代码**：新增 `src/infra/gui/debug_ui.{hpp,cpp}`；不动既有 `gui.hpp` / `imgui_gui.cpp` 的接口；不改 `core/` 任何文件
- **构建**：`src/infra/CMakeLists.txt` 加 1 行 `gui/debug_ui.cpp` 到 `INFRA_SOURCES`；`imgui` 已由 `LX_Infra` 私有链接，无需新依赖
- **测试**：新增 `test_debug_ui_smoke`，注册到 `src/test/CMakeLists.txt`
- **依赖**：`REQ-017`（ImGui overlay 已落地）；`REQ-014`（Clock `deltaTime` / `smoothedDeltaTime`，已存在）
- **下游**：`REQ-019` demo scene viewer 会复用这些 helper；后续材质/后处理 UI 可继续沿用
- **非目标**：不做材质反射编辑器、不做 scene graph inspector、不引入新 GUI framework / DSL / 状态系统、不做截图测试
