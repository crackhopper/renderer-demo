## 1. debug_ui 模块骨架

- [x] 1.1 新建 `src/infra/gui/debug_ui.hpp`：声明 `namespace LX_infra::debug_ui`，包含全部 R2–R4 的函数原型；`.hpp` 仅 include `core/math/vec.hpp`、`core/utils/string_table.hpp`、`core/time/clock.hpp`、`core/scene/camera.hpp`、`core/scene/light.hpp` 与 `<string>`，严禁引入 ImGui 头
- [x] 1.2 新建 `src/infra/gui/debug_ui.cpp`：include `<imgui.h>` + 上述 core 头 + 必要 std；顶部放 `static_assert(sizeof(Vec3f) == 3*sizeof(float))` 与 `static_assert(sizeof(Vec4f) == 4*sizeof(float))`
- [x] 1.3 在 `src/infra/CMakeLists.txt` 的 `INFRA_SOURCES` 里追加 `gui/debug_ui.cpp`

## 2. 基础桥接 helper

- [x] 2.1 在 `.cpp` 实现 `dragVec3` / `dragVec4`：透传 `ImGui::DragFloat3` / `DragFloat4` 到 `value.data`，返回 widget 返回值
- [x] 2.2 实现 `sliderFloat` / `sliderInt`：透传 `ImGui::SliderFloat` / `SliderInt`
- [x] 2.3 实现 `colorEdit3` / `colorEdit4`：透传 `ImGui::ColorEdit3` / `ColorEdit4` 到 `value.data`
- [x] 2.4 实现 `labelText`（`const char*` 与 `std::string` 两个重载）、`labelFloat`、`labelInt`：通过 `ImGui::LabelText` 或 `ImGui::Text` 渲染 `"label: value"` 风格，保持与 `ImGui::LabelText` 一致行为
- [x] 2.5 实现 `labelStringId`：用 `LX_core::GlobalStringTable::get().getName(value.id)` 拿名；空字符串时回退为 `"(empty #<id>)"` 占位符

## 3. Panel 与 section 容器

- [x] 3.1 实现 `beginPanel(title)`：`SetNextWindowPos({8,8}, FirstUseEver)` + `SetNextWindowSize({320,400}, FirstUseEver)` + `ImGui::Begin(title)`，返回 `bool`
- [x] 3.2 实现 `endPanel()`：无条件 `ImGui::End()`
- [x] 3.3 实现 `beginSection(title)` / `endSection()`：基于 `ImGui::CollapsingHeader`；`endSection` 做 no-op 但保留对称
- [x] 3.4 实现 `separatorText(label)`：透传 `ImGui::SeparatorText`

## 4. 内置 panel

- [x] 4.1 实现 `renderStatsPanel(const Clock&)`：至少显示 `frameCount`、`deltaTime`（毫秒）、从 `smoothedDeltaTime` 推导的 FPS（`smoothed > 0 ? 1.0f/smoothed : sentinel`）
- [x] 4.2 实现 `cameraPanel(title, Camera&)`：对 `position` / `target` / `up` 用 `dragVec3`，`fovY` / `aspect` / `nearPlane` / `farPlane` 用 `sliderFloat`（范围给合理默认）；不调用 `updateMatrices()`
- [x] 4.3 实现 `directionalLightPanel(title, DirectionalLight&)`：用 `dragVec4` 编辑 `ubo->param.dir`，`colorEdit4` 编辑 `ubo->param.color`；任一 widget 返回 `true` 时调用 `light.ubo->setDirty()`

## 5. 不做的事（显式拒绝）

- [x] 5.1 确认 `debug_ui.hpp` 不含 `materialPanel` 符号（grep 验证 0 匹配）
- [x] 5.2 确认 `debug_ui.hpp` / `debug_ui.cpp` 都不引入新的 GUI framework、DSL 或 panel 描述器（只有自由函数声明，无类/结构/DSL）

## 6. 集成测试

- [x] 6.1 新建 `src/test/integration/test_debug_ui_smoke.cpp`：链接级验证（把每个公开 helper 的函数指针塞进 `std::vector<void*>`，断言全非空）
- [x] 6.2 在同一测试里加一段 CPU-only ImGui smoke：`ImGui::CreateContext()` → 伪造最小 IO + 构建字体 atlas → `ImGui::NewFrame()` → 依次调 `beginPanel` / `dragVec3` / `colorEdit4` / `labelStringId` / `renderStatsPanel(clock)` / `cameraPanel(...)` / `directionalLightPanel(...)` → `ImGui::EndFrame()` → `ImGui::DestroyContext()`
- [x] 6.3 在 `src/test/CMakeLists.txt` 把 `test_debug_ui_smoke` 加入 `TEST_INTEGRATION_EXE_LIST`，并 `target_link_libraries(test_debug_ui_smoke PRIVATE imgui)` 拿到 ImGui 头
- [x] 6.4 本地构建并运行：`cmake --build build --target test_debug_ui_smoke && ./build/src/test/test_debug_ui_smoke`（`LD_LIBRARY_PATH` 指向 SDL3 build 目录），断言通过

## 7. 收尾

- [x] 7.1 `cmake --build build` 全量构建无回归；`test_input_state` / `test_sdl_input` / `test_imgui_overlay` / `test_debug_ui_smoke` / `test_engine_loop` 用例均 PASS
- [x] 7.2 更新 `docs/requirements/018-debug-panel-helper.md` 的 `## 实施状态` 段落
