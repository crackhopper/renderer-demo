# REQ-019: `demo_scene_viewer` 集成入口

## 背景

当前代码已经把 demo 需要的几块底座铺开了大半：

- `REQ-010` 定义了资产目录与 `cdToWhereAssetsExist(...)`
- `REQ-011` 对应的 `GLTFLoader` 与集成测试已经实际存在，当前代码可以真实读取 `DamagedHelmet.gltf`
- `REQ-012` / `REQ-013` 提供了输入抽象和 SDL3 输入实现
- `REQ-014` / `REQ-015` / `REQ-016` 提供了时钟和两种相机控制器
- `REQ-017` 的 ImGui overlay 已经实际接进当前 VulkanRenderer
- `REQ-018` 的 debug panel helper 仍停留在需求文档阶段
- `REQ-020` 的 `EngineLoop` 已经存在并可用

但这些能力目前还没有被收敛到一个真正面向人工调试的可运行入口中。

当前仓库状态（2026-04-17 核查）：

- [src/main.cpp](../../src/main.cpp) 目前仍然只是 `int main() { return 0; }`
- [src/core/gpu/engine_loop.hpp](../../src/core/gpu/engine_loop.hpp) / [src/core/gpu/engine_loop.cpp](../../src/core/gpu/engine_loop.cpp) 已提供正式主循环骨架
- [src/backend/vulkan/vulkan_renderer.hpp](../../src/backend/vulkan/vulkan_renderer.hpp) / [src/backend/vulkan/vulkan_renderer.cpp](../../src/backend/vulkan/vulkan_renderer.cpp) 已支持 ImGui overlay，并通过 `setDrawUiCallback(...)` 暴露 UI 注入点
- [src/infra/window/sdl_window.cpp](../../src/infra/window/sdl_window.cpp) 已在 SDL 事件循环里把事件转发给 ImGui 和 `Sdl3InputState`
- [src/infra/mesh_loader/gltf_mesh_loader.hpp](../../src/infra/mesh_loader/gltf_mesh_loader.hpp) / [src/infra/mesh_loader/gltf_mesh_loader.cpp](../../src/infra/mesh_loader/gltf_mesh_loader.cpp) 已支持读取几何流、可选 tangent 和最小 PBR 材质元数据
- [src/test/integration/test_gltf_loader.cpp](../../src/test/integration/test_gltf_loader.cpp) 已覆盖 DamagedHelmet 加载、缺失文件和损坏文件路径
- `OrbitCameraController` / `FreeFlyCameraController` 已有实际实现
- [src/test/test_render_triangle.cpp](../../src/test/test_render_triangle.cpp) 已经存在一个“非正式集成入口”，把 `Window + VulkanRenderer + Scene + EngineLoop + Orbit/FreeFly` 串起来了，但它仍是测试程序，不是 `src/demos/scene_viewer/` 形式的正式 demo
- 仓库里还没有 `src/demos/` 目录，也没有默认 scene viewer demo
- `REQ-018` 的 `debug_ui` helper 仍只有需求文档，没有对应实现代码

因此，本需求的职责不是“再定义一套循环”，而是把已经存在的运行时能力接入一个默认的 scene viewer，作为 Phase 1 之后的人工调试入口。

这个 demo 的定位是：

- 不是 CI 测试
- 不是 tutorial 示例
- 是后续渲染能力联调的默认 playground

## 目标

1. 新建 `demo_scene_viewer`，作为默认集成 demo
2. 基于 `EngineLoop` 运行，而不是手写裸 `while` 主循环
3. 基于当前已存在的 `GLTFLoader` 能力，加载 `DamagedHelmet`
4. 默认启用 Orbit，相机可切换到 FreeFly
5. 接入当前已经落地的 ImGui overlay
6. 用最少的 demo glue 把 glTF 几何和当前材质系统桥起来
7. 作为后续 Sponza / shadow / IBL / post-process 的扩展基底

## 需求

### R1: 新建 demo 目录

新增：

```text
src/demos/
├── CMakeLists.txt
└── scene_viewer/
    ├── CMakeLists.txt
    ├── main.cpp
    └── README.md
```

说明：

- `scene_viewer` 是本仓库第一个正式 demo
- 后续若新增更多 demo，继续放在 `src/demos/` 下
- demo 不进入 `src/test/`

### R2: 运行骨架必须基于 `EngineLoop`

`REQ-020` 已经完成，因此本 REQ 明确要求 `demo_scene_viewer` 通过 `EngineLoop` 运行。

主流程至少包含：

1. 调用 `cdToWhereAssetsExist("models/damaged_helmet/DamagedHelmet.gltf")`
2. 创建 `LX_infra::Window`
3. 创建 `LX_core::backend::VulkanRenderer`
4. 调用 `renderer->initialize(window, "demo_scene_viewer")`
5. 创建并填充 `Scene`
6. 创建 `EngineLoop`
7. `loop.initialize(window, renderer)`
8. `loop.startScene(scene)`
9. 注册 update hook
10. `loop.run()`

约束：

- 不再把 demo 文档写成裸 `while (running) { uploadData(); draw(); }`
- 时间读取以 update hook 传入的 `Clock` 为准；`EngineLoop::getClock()` 可作为辅助读取接口
- 业务更新逻辑通过 `EngineLoop::setUpdateHook(...)` 接入

### R3: 当前 `GLTFLoader` 输入契约

当前代码里的 `GLTFLoader` 已经提供了本 demo 可直接消费的最小输入契约：

- `infra::GLTFLoader::load("assets/models/damaged_helmet/DamagedHelmet.gltf")`
- `getPositions()`
- `getNormals()`
- `getTexCoords()`
- `getIndices()`
- `getTangents()`
- `getMaterial()`

其中 `getMaterial()` 返回的最小 PBR 元数据至少包含：

- `baseColorFactor`
- `metallicFactor`
- `roughnessFactor`
- `emissiveFactor`
- `baseColorTexture`
- `metallicRoughnessTexture`
- `normalTexture`
- `occlusionTexture`
- `emissiveTexture`

说明：

- `GLTFLoader` 自身更细的行为定义仍属于 `REQ-011`
- 本 REQ 只定义 scene viewer 如何消费这些输出

### R4: 资产与场景基线

首版 `demo_scene_viewer` 的默认场景至少包含：

1. `DamagedHelmet`
2. 一块地面 plane
3. 默认方向光
4. `Scene` 中的一台可控制相机

资产定位要求：

- 必须先调用 `cdToWhereAssetsExist("models/damaged_helmet/DamagedHelmet.gltf")`
- 然后以 `assets/models/damaged_helmet/DamagedHelmet.gltf` 作为 glTF 路径

首版不要求加载 `Sponza`。`Sponza` 是后续扩展目标，不是本 REQ 的首批验收必选项。

### R5: glTF 几何与材质桥接属于 demo glue

当前正式材质入口仍是 [src/infra/material_loader/generic_material_loader.hpp](../../src/infra/material_loader/generic_material_loader.hpp) 提供的 `LX_infra::loadGenericMaterial(...)`，而不是完整的 glTF/PBR 材质系统。

因此本 demo 允许包含少量过渡 glue：

1. `buildMeshFromGltf(loader)`
2. `makeHelmetMaterial(loader.getMaterial(), gltfDir)`
3. 必要时的 `makeGroundPlane()` / `makeGroundMaterial()`

具体要求：

- `buildMeshFromGltf(loader)` 负责把 `GLTFLoader` 输出拼成当前工程可消费的 mesh / vertex buffer / index buffer
- 若 glTF 缺少 tangent，而当前 demo 材质路径又需要 tangent，可用受控的占位值；不要把 tangent 生成算法塞进本 REQ
- `makeHelmetMaterial(...)` 负责调用 `LX_infra::loadGenericMaterial(...)` 加载现有 `.material` 资产，并尽量把 `GLTFPbrMaterial` 的贴图路径桥接到当前材质绑定
- 当前材质系统不支持的 glTF 字段可以跳过

约束：

- 这些 helper 属于 demo glue，不要求沉入 `infra/`
- 本 REQ 不要求实现完整 PBR material loader
- demo 内部桥接逻辑可以先围绕 `DamagedHelmet`，不追求完全泛化

### R6: Renderable 路径以 `SceneNode` 为主

当前仓库同时存在：

- `SceneNode`：推荐主路径
- `RenderableSubMesh`：仍存在，但 [REQ-024](024-remove-renderable-submesh-legacy-abstraction.md) 已将其定义为 legacy

因此本 REQ 的要求是：

- demo 应优先使用 `SceneNode`
- 文档与示例不得再把 `RenderableSubMesh` 写成长期推荐对象模型
- 若实现初期因为 glue 限制不得不短暂借用 `RenderableSubMesh`，只能视为临时过渡，不得写入本需求的推荐路径

### R7: 相机交互必须贴合当前控制器实现

`demo_scene_viewer` 至少接入两种控制器：

1. `OrbitCameraController`
2. `FreeFlyCameraController`

交互要求：

- 默认模式为 Orbit
- 按 `F2` 在 Orbit / FreeFly 间切换
- 相机每帧在 update hook 中更新
- controller 更新后，调用方负责执行 `camera.updateMatrices()`

基于当前控制器实现，Help panel 至少要反映以下默认操作：

- Orbit：
  - 鼠标左键拖拽旋转
  - 鼠标右键拖拽平移 target
  - 鼠标滚轮缩放
- FreeFly：
  - 鼠标右键按住时视角转动
  - `W/A/S/D` 平移
  - `Space` 上升
  - `LShift` 下降
  - `LCtrl` 加速

说明：

- `Sdl3InputState` 当前只提供状态量，不提供边沿事件
- 因此 `F1` / `F2` 这种切换键的边沿检测允许在 demo 中手写“上一帧 vs 当前帧”的局部逻辑

### R8: ImGui 接入必须复用当前 VulkanRenderer overlay

当前代码里，ImGui overlay 已经接进 `VulkanRenderer`，而不是挂在 `gpu::Renderer` 抽象基类上。

因此本 REQ 要求：

- demo 必须使用 `LX_core::backend::VulkanRenderer`
- UI 绘制通过 `VulkanRenderer::setDrawUiCallback(std::function<void()>)` 注入
- 不能假设 `gpu::Renderer` 基类已经统一暴露 UI callback API

demo 中的 UI 至少包括：

1. Render Stats panel
2. Camera panel
3. Directional Light panel
4. Help panel

Help panel 至少说明：

- `F1`：显示/隐藏帮助
- `F2`：切换 Orbit / FreeFly
- Orbit 的鼠标操作
- FreeFly 的鼠标/键盘操作

关于 `REQ-018`：

- 若 `REQ-018` 在实现本 REQ 前已落地，则应优先复用 `LX_infra::debug_ui::*`
- 若 `REQ-018` 仍未落地，则允许 `scene_viewer` 先在 demo 目录内自带最小 UI helper
- 本 REQ 不再把 `REQ-018` 视为硬阻塞前置

另外，当前仓库中的 [src/test/test_render_triangle.cpp](../../src/test/test_render_triangle.cpp) 已经证明：

- `EngineLoop`
- `VulkanRenderer`
- SDL 输入
- Orbit / FreeFly 切换

这套链路可以被串起来运行。

但它仍然不是本 REQ 的完成态，因为它：

- 仍位于测试程序路径
- 使用的是手写三角形而不是 `DamagedHelmet`
- 仍沿用 `RenderableSubMesh`
- 没有正式的 scene viewer UI 结构
- 没有 `src/demos/scene_viewer/README.md`

### R9: SDL backend 是首版主线

基于当前代码状态，首版 scene viewer 主线以 SDL backend 为准。

原因：

- `sdl_window.cpp` 已把 SDL 事件转发给 ImGui 和输入状态
- `Gui::init(...)` 当前走的是 SDL3 + Vulkan backend
- `test_imgui_overlay` 也围绕 SDL 路径验证 overlay 生命周期

因此：

- 首版 demo 以 `USE_SDL=ON` 为主线配置
- GLFW 路径不是本 REQ 的主验收路径

### R10: CMake 接入

新增：

- `src/demos/CMakeLists.txt`
- `src/demos/scene_viewer/CMakeLists.txt`

顶层 [CMakeLists.txt](../../CMakeLists.txt) 需要接入 `src/demos/`，建议方式如下：

```cmake
option(LX_BUILD_DEMOS "Build demo executables" ON)

if(LX_BUILD_DEMOS)
  add_subdirectory(src/demos)
endif()
```

说明：

- 当前仓库没有 `src/CMakeLists.txt`，因此不能把 demo 注册写到不存在的路径里
- `demo_scene_viewer` 不属于 ctest
- demo target 应链接 `${CORE_LIB}`、`${INFRA_LIB}`、`${GRAPHICS_LIB}`
- 顶层现有 `Renderer` 可执行文件是否继续保留占位 `src/main.cpp`，不由本 REQ 强制决定；但本 REQ 至少要新增独立 demo target

### R11: README

`src/demos/scene_viewer/README.md` 至少包含：

1. demo 目的
2. 依赖的前置需求
3. 构建与运行方式
4. 控制说明
5. 已知限制

已知限制至少包括：

- 当前材质桥接仍是过渡方案，不是完整 PBR
- ImGui 当前是 overlay，不在 FrameGraph 中
- 首版以 SDL 路径为准
- 首版场景以 `DamagedHelmet` 为主，不强制包含 `Sponza`

## 验收

本 REQ 不要求自动化测试；验收以人工运行为主。

最小验收清单：

1. 能成功启动 `demo_scene_viewer`
2. 能看到 `DamagedHelmet` 和地面
3. Orbit 模式可正常旋转 / 平移 / 缩放
4. `F2` 可切换到 FreeFly，且移动正常
5. ImGui panel 正常显示
6. Camera / Light 面板修改后，画面会产生可见变化
7. 关闭窗口时无崩溃

## 修改范围

| 文件 | 改动 |
|---|---|
| `src/demos/CMakeLists.txt` | 新增 |
| `src/demos/scene_viewer/CMakeLists.txt` | 新增 |
| `src/demos/scene_viewer/main.cpp` | 新增 |
| `src/demos/scene_viewer/README.md` | 新增 |
| `CMakeLists.txt` | 新增 `LX_BUILD_DEMOS` 选项并接入 `src/demos` |

## 边界与约束

- 不要求自动化测试
- 不要求完整 PBR 材质系统
- 不要求首版支持 `Sponza`
- 不要求引入 scene 序列化、热重载、编辑器框架
- `main.cpp` 应保持可读；若 glue 过多，拆到 `scene_viewer/` 子文件中
- 主线以 SDL backend 为准
- 不把 `REQ-018` 作为硬阻塞；缺少统一 helper 时可先用 demo 内部最小 UI 实现

## 依赖

- `REQ-010`：资产目录与 `cdToWhereAssetsExist()`
- `REQ-011`：当前 `GLTFLoader` 契约与 DamagedHelmet 加载闭环
- `REQ-012`：输入抽象
- `REQ-013`：SDL 真实输入
- `REQ-014`：`Clock`
- `REQ-015`：Orbit 控制器
- `REQ-016`：FreeFly 控制器
- `REQ-017`：ImGui overlay
- `REQ-020`：`EngineLoop` 已完成并作为本 REQ 的正式运行骨架

`REQ-018` 不是硬依赖；若已完成则优先复用，否则允许本 REQ 自带最小 UI glue。

## 下游

- 后续 `Sponza` 场景扩展
- shadow / IBL / post-process 等渲染功能集成
- 更完整的材质与场景调试面板
- 若 `REQ-018` 后续落地，可把 scene viewer 内部 UI glue 回收为统一 helper

## 实施状态

2026-04-17 已落地（对应 OpenSpec change `demo-scene-viewer`）：

- 顶层 `CMakeLists.txt` 加 `option(LX_BUILD_DEMOS "Build demo executables" ON)` 并在 ON 时 `add_subdirectory(src/demos)`
- 新建 `src/demos/CMakeLists.txt` 与 `src/demos/scene_viewer/` 目录结构（`CMakeLists.txt` / `main.cpp` / `scene_builder.{hpp,cpp}` / `camera_rig.{hpp,cpp}` / `ui_overlay.{hpp,cpp}` / `README.md`）
- `main.cpp` 基于 `EngineLoop::run()` 运行（非裸 while 主循环）；启动顺序 `cdToWhereAssetsExist` → `LX_infra::Window` → `VulkanRenderer` → `Scene`（helmet + ground + 默认方向光） → `EngineLoop::initialize/startScene/setUpdateHook` → `setDrawUiCallback` → `run`
- `scene_builder` 实现 `buildMeshFromGltf`（`VertexPosNormalUvBone` 拼装；TANGENT 缺失占位 `{1,0,0,1}` 并 warning，不做 MikkTSpace 生成）、`makeHelmetMaterial`（基于 `materials/blinnphong_default.material` 桥接 glTF `baseColorTexture` → `albedoMap` 绑定，`enableAlbedo=1` / `enableNormal=0`；其余 PBR 贴图未绑定）、`buildHelmetNode` / `buildGroundNode`（20m×20m XZ 平面）
- `camera_rig` 把 Orbit / FreeFly + `F2` 边沿切换 + pose 同步（`syncOrbitFromCamera` / `syncFreeFlyFromCamera` 与 `test_render_triangle.cpp` 公式一致）封成 demo-local rig；Orbit 为默认模式，`camera.updateMatrices()` 由 rig 调用
- `ui_overlay` 通过 `VulkanRenderer::setDrawUiCallback` 注入；Stats / Camera / Directional Light 三面板复用 `LX_infra::debug_ui` helper（REQ-018），Help 面板 demo-local，`F1` 边沿切换显示
- `src/demos/scene_viewer/README.md` 含 Purpose / Upstream requirements / Build & run（SDL3 LD_LIBRARY_PATH 提示）/ Controls / Known limitations 五段，并复述 7 项手工验收 checklist
- `cmake --build build --target demo_scene_viewer` 本地构建通过，无回归（`test_input_state` / `test_sdl_input` / `test_imgui_overlay` / `test_debug_ui_smoke` / `test_gltf_loader` / `test_engine_loop` / `test_assets_layout` 全部 PASS）
- headless 环境下确认 `cdToWhereAssetsExist` 成功且 SDL video device 不存在时早失败（exit code 2）；窗口打开、4 块 panel 显示、F2 模式切换、Camera/Light 编辑可见变化、关闭无崩溃——这 7 项人工验收需用户在有显示环境执行（REQ 明确不要求自动化）

本次核查结论：代码与构建闭环已成立，但因当前环境无法完成显示输出下的人工验收，剩余收尾统一并入 [`REQ-034`](034-remaining-validated-backlog.md)。
- `src/main.cpp` 保持 `int main() { return 0; }` 占位；`src/test/test_render_triangle.cpp` 保留不动
