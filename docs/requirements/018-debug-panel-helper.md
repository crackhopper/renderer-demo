# REQ-018: DebugPanel helper（封装常用 ImGui widget）

## 背景

REQ-017 把 ImGui 接进 `VulkanRenderer` 后，上层 demo 就可以在 `setDrawUiCallback` 里直接调 `ImGui::Begin / SliderFloat / ColorEdit3 / End`。但每个 demo 都从原始 ImGui API 开始重写"显示 FPS"、"调整 light 方向"、"显示 camera 状态"是重复劳动。

更深层的问题：

- ImGui API 直接操作裸 float 指针、std::string 与之不兼容、`ImGui::SliderFloat3` 接 `float[3]` 而引擎里是 `Vec3f` —— 每次都要 reinterpret_cast 或临时 buffer
- demo 想"显示一个相机的所有字段"必须列 6-7 行 ImGui 调用，无法复用
- 没有约定的 panel 命名 / 组织风格 → 多个 demo 长得不一致
- 后续 Phase 2 的 `describe()` / dump API 想接入 UI 需要一个稳定的 helper 层

本 REQ 引入一个**轻量** helper namespace `LX_core::debug_ui`（**不是**新 GUI 框架），把常用渲染调试 widget 封装成几个一行调用。helper 直接调 ImGui，不引入第三方依赖。

## 目标

1. 提供 `Vec3f` / `Vec4f` / `StringID` 等引擎类型与 ImGui widget 的桥接函数
2. 提供针对 `Camera` / `LightBase` / `Clock` / `Material` 的"一行 panel"
3. 提供一个 `DebugPanel::beginPanel(title)` / `endPanel()` 风格统一外观
4. 不持有任何状态 —— 全部纯函数 + 显式 in/out 参数
5. helper 文件归属在 **infra** 层（不进 core），因为它直接调 ImGui

## 需求

### R1: 类型桥接函数

新建 `src/infra/gui/debug_ui.hpp` + `.cpp`：

```cpp
#pragma once
#include "core/math/vec.hpp"
#include "core/string/string_id.hpp"
#include <string>

namespace LX_infra::debug_ui {

// 基础 widget —— 引擎类型 ↔ ImGui

/// 等价 ImGui::DragFloat3，但接 Vec3f &
bool dragVec3(const char* label, LX_core::Vec3f& v,
              float speed = 0.01f, float min = 0.0f, float max = 0.0f);

bool dragVec4(const char* label, LX_core::Vec4f& v,
              float speed = 0.01f, float min = 0.0f, float max = 0.0f);

bool sliderFloat(const char* label, float& v, float min, float max);
bool sliderInt(const char* label, int& v, int min, int max);

/// RGB 颜色编辑（带 picker）
bool colorEdit3(const char* label, LX_core::Vec3f& rgb);
bool colorEdit4(const char* label, LX_core::Vec4f& rgba);

/// 只读字段
void labelText(const char* label, const std::string& value);
void labelInt(const char* label, int value);
void labelFloat(const char* label, float value);
void labelStringId(const char* label, LX_core::StringID id);

}
```

实现一律是 ImGui API 的 thin wrapper。`Vec3f` 取地址转 `float*` 安全因为 `Vec3f` 是 plain struct（按需在实现里 `static_assert(sizeof(Vec3f) == 12)`）。

### R2: Panel 容器

```cpp
namespace LX_infra::debug_ui {

/// 等价 ImGui::Begin，但带项目统一风格（位置、大小默认值）。
/// 返回 false 时调用方应该跳过内容、依然要调 endPanel()。
bool beginPanel(const char* title);
void endPanel();

/// 折叠组（CollapsingHeader）
bool beginGroup(const char* title);
void endGroup();

/// 简短的两列布局（label : value）
void twoColumnRow(const char* label, const char* value);

}
```

`beginPanel` 内部：

```cpp
ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
ImGui::SetNextWindowSize(ImVec2(300, 0), ImGuiCond_FirstUseEver);
return ImGui::Begin(title, nullptr,
                    ImGuiWindowFlags_AlwaysAutoResize);
```

### R3: 引擎对象 panel

针对 `Camera` / `LightBase` / `Clock`：

```cpp
namespace LX_infra::debug_ui {

/// 渲染统计 panel：FPS / frameCount / deltaTime
void renderStatsPanel(const LX_core::Clock& clock);

/// Camera 调试 panel：position / target / fov / aspect / near / far
/// 字段全部可编辑（绑定到 camera 字段地址），用户改完调用方负责 updateMatrices。
void cameraPanel(const char* label, LX_core::Camera& camera);

/// DirectionalLight 调试 panel：direction / color / intensity
/// 修改后会自动 setDirty（写入 ubo->setDirty()）
void directionalLightPanel(const char* label, LX_core::DirectionalLight& light);

}
```

实现示例（renderStatsPanel）：

```cpp
void renderStatsPanel(const LX_core::Clock& clock) {
  if (!beginPanel("Render Stats")) { endPanel(); return; }
  const float fps = 1.0f / std::max(clock.smoothedDeltaTime(), 1e-5f);
  labelFloat("FPS", fps);
  labelFloat("dt (ms)", clock.deltaTime() * 1000.0f);
  labelInt("frame", static_cast<int>(clock.frameCount()));
  endPanel();
}
```

`cameraPanel` 示例：

```cpp
void cameraPanel(const char* label, LX_core::Camera& camera) {
  if (!beginPanel(label)) { endPanel(); return; }
  dragVec3("position", camera.position, 0.05f);
  dragVec3("target",   camera.target,   0.05f);
  sliderFloat("fovY",  camera.fovY,     5.0f, 120.0f);
  sliderFloat("near",  camera.nearPlane, 0.01f, 10.0f);
  sliderFloat("far",   camera.farPlane,  10.0f, 1000.0f);
  endPanel();
}
```

### R4: Material panel（最小版）

```cpp
/// 列出 material 的所有 uniform 字段，根据反射类型自动选 widget。
/// 走 IShader / ShaderResourceBinding 反射结果。Phase 1 暂只支持 float / vec3 / vec4。
void materialPanel(const char* label, LX_core::Material& mat);
```

实现要点：

- 拿到 `mat.getShader()->getResourceBinding()`（已有的反射 API）
- 遍历 UBO member，按 type dispatch：
  - `float` → `sliderFloat`（默认范围 0..1，可在 Material 上加 range hint，或者本 REQ 写死）
  - `Vec3f` → `colorEdit3`（如果 member 名包含 "color" / "albedo" / "emissive"）否则 `dragVec3`
  - `Vec4f` → `colorEdit4`
- 不支持的 type 显示 `labelText` 占位

注意：`materialPanel` 是一个 **best effort** helper，复杂材质（如 PBR 多通道贴图开关）需要 demo 自己写专用 panel；本 REQ 只覆盖最常见的 80% 路径。

### R5: 集成测试

由于所有 helper 都直接调 ImGui，单元测试只能跑非常基础的"接口存在 + 编译通过"测试。

新建 `src/test/integration/test_debug_ui_smoke.cpp`：

```cpp
TEST(DebugUi, link_check) {
  // 不真正调用（需要 ImGui context），只验证符号存在
  using namespace LX_infra::debug_ui;
  auto p1 = &dragVec3;
  auto p2 = &renderStatsPanel;
  auto p3 = &cameraPanel;
  EXPECT_NE(p1, nullptr);
  EXPECT_NE(p2, nullptr);
  EXPECT_NE(p3, nullptr);
}
```

真正的功能验证依赖 REQ-019 的人工跑 demo。

## 修改范围

| 文件 | 改动 |
|---|---|
| `src/infra/gui/debug_ui.hpp` | 新增 |
| `src/infra/gui/debug_ui.cpp` | 新增 |
| `src/infra/gui/CMakeLists.txt` | 把新文件加进 sources |
| `src/test/integration/test_debug_ui_smoke.cpp` | 新增 |
| `src/test/integration/CMakeLists.txt` | 注册 |

## 边界与约束

- **不引入** 任何新依赖
- **不**自创"GUI DSL"或"reactive panel" —— 直接调 ImGui，调用方完全保留 ImGui 原生 API 的访问
- **不持有状态** —— 所有 helper 都是纯函数。如果未来需要"开/关 panel" 这类持久状态，由 demo 自己 own 一个 bool
- `materialPanel` 只覆盖 float / vec3 / vec4，不支持 mat4 / texture handle / int array
- 不试图做"自动 capability dump" —— Phase 2 REQ-209 / REQ-210 的 `describe()` 可以喂给本 helper，但本 REQ 不绑定它
- 命名空间用 `LX_infra::debug_ui` 而不是 `LX_core::*` —— 因为 helper 直接 include ImGui header，core 层不应该 leak GUI 依赖

## 依赖

- **REQ-017**（必需）：ImGui 已经在 `VulkanRenderer::draw` 里被调用且 context 可用
- 不强依赖 REQ-014 —— 但 `renderStatsPanel(Clock&)` 需要 `Clock`，所以**事实上**需要 REQ-014 已经合入

## 下游

- **REQ-019**：demo_scene_viewer 在 `setDrawUiCallback` 内调本 helper
- 后续 Phase 1 REQ-101+：post-process 调试面板（exposure / bloom threshold / FXAA toggle）走同一组 helper
- **Phase 2 REQ-209 / REQ-210**：`describe()` API 输出可以接入 `materialPanel` / `cameraPanel` 的反射路径

## 实施状态

2026-04-16 核查结果：未开始。
