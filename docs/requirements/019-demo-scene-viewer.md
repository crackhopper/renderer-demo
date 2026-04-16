# REQ-019: `demo_scene_viewer` 集成入口

## 背景

`REQ-010` 到 `REQ-018` 分别定义了资产目录、glTF 读取、输入抽象、SDL 输入、时钟、相机控制器、ImGui overlay 和 debug UI helper，但这些能力目前还没有被收敛到一个真正面向人类调试的可运行入口中。

当前仓库状态（2026-04-16 核查）：

- [src/main.cpp](../../src/main.cpp) 目前只是 `int main() { return 0; }`
- `EngineLoop` 已经实际存在于 [src/core/gpu/engine_loop.hpp](../../src/core/gpu/engine_loop.hpp) / [engine_loop.cpp](../../src/core/gpu/engine_loop.cpp)，`REQ-020` 已完成
- 还没有 `src/demos/` 目录
- 还没有一个同时覆盖资产加载、相机交互、ImGui 调试 UI 的默认 demo

因此，`REQ-019` 的职责不是“再定义一套主循环”，而是把前置能力接入一个基于 `EngineLoop` 的 scene viewer，作为 Phase 1 后续渲染功能开发的默认人工调试入口。

这个 demo 的定位是：

- 不是 CI 测试
- 不是 tutorial 示例
- 是后续渲染功能的默认集成 playground

## 目标

1. 新建 `demo_scene_viewer`，作为默认集成 demo
2. 基于 `EngineLoop` 组织运行，而不是手写裸 `while` 主循环
3. 加载 `DamagedHelmet`，并提供一个最小地面与默认方向光
4. 默认启用 Orbit，相机可切换到 FreeFly
5. 接入 ImGui overlay 与 `debug_ui` helper
6. 作为后续 Sponza / shadow / IBL / post-process 的扩展基底

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

1. 调用 `cdToWhereAssetsExist(...)` 定位资产根
2. 创建 `Window`
3. 创建 `VulkanRenderer`
4. 调用 `renderer->initialize(window, "demo_scene_viewer")`
5. 创建并填充 `Scene`
6. 创建 `EngineLoop`
7. `loop.initialize(window, renderer)`
8. `loop.startScene(scene)`
9. 注册 update hook
10. `loop.run()`

约束：

- 不再把 demo 文档写成裸 `while (running) { uploadData(); draw(); }`
- `Clock` 读取以 `EngineLoop::getClock()` 为准
- 业务更新逻辑通过 `EngineLoop::setUpdateHook(...)` 接入

### R3: 资产与场景基线

首版 `demo_scene_viewer` 的默认场景至少包含：

1. `DamagedHelmet`
2. 一块地面 plane
3. 默认方向光
4. `Scene` 自带默认相机

资产定位要求：

- 必须先调用 `cdToWhereAssetsExist("models/damaged_helmet/DamagedHelmet.gltf")`
- 然后以 `assets/models/damaged_helmet/DamagedHelmet.gltf` 作为加载路径

首版不要求一上来就加载 `Sponza`。`Sponza` 应作为后续扩展目标，而不是本 REQ 的首批验收必选项。

### R4: glTF 几何与材质桥接是 demo 内部 glue

本 REQ 依赖 `REQ-011` 提供：

- `GLTFLoader` 读取 `DamagedHelmet.gltf`
- positions / normals / uv / indices
- 可选 tangent
- 基础 PBR 贴图元数据

但当前正式材质入口仍是 generic material asset loader，而不是完整 PBR 材质系统。因此本 demo 允许包含少量“过渡 glue”：

1. `buildMeshFromGltf(loader)`
- 把 `GLTFLoader` 输出拼成当前工程可消费的 mesh / vertex buffer / index buffer
- 若 tangent 不存在，可用占位值

2. `makeHelmetMaterial(...)`
- 基于现有 `.material` 资产，例如 `materials/blinnphong_default.material` 或同类材质，调用 `LX_infra::loadGenericMaterial(...)`
- 根据 `GLTFPbrMaterial` 中可用的贴图路径，把 baseColor / normal 等资源尽量桥接到当前材质绑定
- 当前材质不支持的字段可以跳过

约束：

- 这些 helper 属于 demo glue，不要求沉入 `infra/`
- 本 REQ 不要求引入真正的 PBR material loader
- demo 内部桥接逻辑可以先服务于 `DamagedHelmet`，不追求完全泛化

### R5: Renderable 路径优先兼容当前主模型

当前仓库同时存在：

- `SceneNode`：推荐主路径
- `RenderableSubMesh`：仍存在，但后续需求已将其标为 legacy

因此本 REQ 的要求是：

- demo 应优先尝试使用 `SceneNode`
- 若在落地过程中因为现有 mesh/material glue 限制需要暂时使用 `RenderableSubMesh`，允许作为过渡实现
- 但文档不得再把 `RenderableSubMesh` 写成长期推荐对象模型

### R6: 相机交互

`demo_scene_viewer` 至少接入两种控制器：

1. `OrbitCameraController`
2. `FreeFlyCameraController`

交互要求：

- 默认模式为 Orbit
- 按 `F2` 在 Orbit / FreeFly 间切换
- 相机每帧在 update hook 中更新
- 调用方负责在 controller 更新后执行 `camera.updateMatrices()`

说明：

- `F1` 可用于显示/隐藏帮助 panel
- 切换键的边沿检测允许在 demo 中手写，不要求先引入新的输入边沿抽象

### R7: ImGui 与 debug UI 接入

本 REQ 依赖：

- `REQ-017`：`VulkanRenderer` 已支持 ImGui overlay 与 `setDrawUiCallback(...)`
- `REQ-018`：已有 `LX_infra::debug_ui::*`

demo 中的 UI 至少包括：

1. Render Stats panel
2. Camera panel
3. Directional Light panel
4. 一个 Help panel

Help panel 至少说明：

- `F1`：显示/隐藏帮助
- `F2`：切换 Orbit / FreeFly
- Orbit 的鼠标操作
- FreeFly 的鼠标/键盘操作

### R8: CMake 接入

新增：

- `src/demos/CMakeLists.txt`
- `src/demos/scene_viewer/CMakeLists.txt`

顶层 [CMakeLists.txt](../../CMakeLists.txt) 需要接入 `src/demos/`，方式应与当前仓库结构一致：

```cmake
option(LX_BUILD_DEMOS "Build demo executables" ON)

if(LX_BUILD_DEMOS)
  add_subdirectory(src/demos)
endif()
```

说明：

- 当前仓库没有 `src/CMakeLists.txt`，因此不能把 demo 注册写到不存在的路径里
- `demo_scene_viewer` 不属于 ctest
- demo target 应链接 `${CORE_LIB}`、`${INFRA_LIB}`、`${GRAPHICS_LIB}` 或与当前顶层构建变量等价的真实库名

### R9: README

`src/demos/scene_viewer/README.md` 至少包含：

1. demo 目的
2. 依赖的前置需求
3. 构建与运行方式
4. 控制说明
5. 已知限制

已知限制至少包括：

- 当前材质桥接仍是过渡方案，不是完整 PBR
- ImGui 当前是 overlay，不在 FrameGraph 中
- GLFW 路径不是主线；首版以 SDL 为准
- 首版场景以 `DamagedHelmet` 为主，不强制包含 `Sponza`

## 验收

本 REQ 不要求自动化测试；验收以人工运行为主。

最小验收清单：

1. 能成功启动 `demo_scene_viewer`
2. 能看到 `DamagedHelmet` 和地面
3. Orbit 模式可正常旋转/缩放/平移
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

## 依赖

- `REQ-010`：资产目录与 `cdToWhereAssetsExist()`
- `REQ-011`：`GLTFLoader` 可读取 `DamagedHelmet`
- `REQ-012`：输入抽象
- `REQ-013`：SDL 真实输入
- `REQ-014`：`Clock`
- `REQ-015`：Orbit 控制器
- `REQ-016`：FreeFly 控制器
- `REQ-017`：ImGui overlay
- `REQ-018`：`debug_ui` helper
- `REQ-020`：`EngineLoop` 已完成并作为本 REQ 的正式运行骨架

## 下游

- 后续 `Sponza` 场景扩展
- shadow / IBL / post-process 等渲染功能集成
- 更完整的材质与场景调试面板

## 实施状态

2026-04-16 核查结果：未开始。

- 当前只有占位的 [src/main.cpp](../../src/main.cpp)
- 还没有 `src/demos/`
- `demo_scene_viewer` 尚不存在
