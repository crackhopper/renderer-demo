# Requirements Index

本文档汇总 `docs/requirements/` 下当前活跃需求的概要，并给出建议执行顺序。

## 说明

- `finished/` 中的需求已经归档，不再列入主执行顺序。
- 当前推荐顺序以 `REQ-010` 到 `REQ-029` 为主。
- `REQ-031` 之后的需求当前不纳入本轮执行排序；如果后续需要再单独规划。

## 当前需求概要

| REQ | 主题 | 当前判断 | 说明 |
|---|---|---|---|
| `REQ-010` | 测试资产与 `assets/` 目录约定 | 未开始 | 先把资产入口、目录结构和 helper 定下来 |
| `REQ-011` | glTF PBR loader | 未开始 | 依赖 `REQ-010` 提供测试资产 |
| `REQ-012` | 输入抽象接口 | 未开始 | 相机控制和 ImGui 输入协调的接口前置 |
| `REQ-013` | SDL3 输入实现 | 未开始 | 把 `REQ-012` 落到真实窗口事件循环 |
| `REQ-014` | Clock 收尾 | 部分完成 | `Clock` 已有主体实现，还差平滑 dt 和测试 |
| `REQ-015` | Orbit 相机控制器 | 未开始 | 依赖输入抽象 |
| `REQ-016` | FreeFly 相机控制器 | 未开始 | 依赖输入抽象和 `Clock` |
| `REQ-017` | ImGui overlay 接入 | 未开始 | 依赖输入与窗口事件接线 |
| `REQ-018` | DebugPanel helper | 未开始 | 依赖 ImGui overlay |
| `REQ-019` | `demo_scene_viewer` | 未开始 | 集成入口，依赖前面基础设施 |
| `REQ-024` | 移除 `RenderableSubMesh` legacy 抽象 | 部分完成 | `SceneNode` 主路径已成立，剩余是删除 legacy 类型和迁移调用点 |
| `REQ-025` | 自定义材质模板与 loader 契约 | 大部分完成 | 通用 `.material` loader 已存在，剩余是补更扎实的示例 |
| `REQ-026` | Camera layer / culling mask | 未开始 | 独立功能，可放在 demo 基础稳定后 |
| `REQ-027` | SpotLight | 前置已具备 | `LightBase` / `Scene` 多 light 容器已就绪，缺具体类型 |
| `REQ-028` | IBL 环境光资源接入 | 未开始 | 依赖环境贴图加载和预过滤能力 |
| `REQ-029` | 多光源 scene-level 资源模型 | 前置已具备 | `Scene` 已支持多个 light object，缺统一 GPU light set 合同 |

## 推荐执行顺序

### 第一阶段：补齐基础入口

1. [`REQ-010`](010-test-assets-and-layout.md)：先定资产目录、测试资产和 `cdToWhereAssetsExist()`，这是后续 demo 和 glTF 的共同基础。
2. [`REQ-012`](012-input-abstraction.md)：先把输入接口立住，避免后续相机和 ImGui 各自拉一套输入路径。
3. [`REQ-014`](014-clock-and-delta-time.md)：把现有 `Clock` 收尾，补 `smoothedDeltaTime()` 和测试。
4. [`REQ-013`](013-sdl-input-impl.md)：在 `REQ-012` 之上接 SDL3 真实输入。

### 第二阶段：形成可交互调试能力

5. [`REQ-015`](015-orbit-camera-controller.md)：先做 Orbit，最适合模型查看。
6. [`REQ-016`](016-freefly-camera-controller.md)：再做 FreeFly，用于大场景漫游。
7. [`REQ-017`](017-imgui-overlay.md)：接入 ImGui overlay，形成基本调试 UI 容器。
8. [`REQ-018`](018-debug-panel-helper.md)：在 ImGui 基础上补统一 debug panel helper。

### 第三阶段：形成完整调试入口

9. [`REQ-011`](011-gltf-pbr-loader.md)：在资产目录已经稳定后做 glTF loader。
10. [`REQ-019`](019-demo-scene-viewer.md)：把资产、输入、相机、ImGui、Clock 全部串起来，形成默认调试入口。

### 第四阶段：整理现有结构债务

11. [`REQ-024`](024-remove-renderable-submesh-legacy-abstraction.md)：在 demo 和主路径稳定后，迁移测试和示例，删除 `RenderableSubMesh`。
12. [`REQ-025`](025-custom-material-template-and-loader.md)：补完示例和文档，完成这一条的最终收尾。

### 第五阶段：光照与可见性扩展

13. [`REQ-026`](026-camera-visibility-layer-mask.md)：给多 camera / debug view 做 layer 过滤能力。
14. [`REQ-027`](027-spot-light.md)：在现有 `LightBase` 框架上补聚光灯类型。
15. [`REQ-029`](029-multi-light-scene-resource-model.md)：在 `SpotLight` 和多 light object 容器基础上，定义统一 multi-light GPU 合同。
16. [`REQ-028`](028-ibl-environment-lighting.md)：等环境贴图资源链路具备后，再把 IBL 作为 scene-level 资源接进来。

## 依赖关系速记

- `REQ-011` 依赖 `REQ-010`
- `REQ-013` 依赖 `REQ-012`
- `REQ-015` 依赖 `REQ-012`
- `REQ-016` 依赖 `REQ-012` 和 `REQ-014`
- `REQ-017` 依赖 `REQ-012` 和 `REQ-013`
- `REQ-018` 依赖 `REQ-017`
- `REQ-019` 依赖 `REQ-010` 到 `REQ-018` 的主要交付项
- `REQ-027` 是 `REQ-029` 的前置之一
- `REQ-028` 依赖环境贴图加载 / 预过滤能力，不建议提前于资源链路落地

## 备注

如果下一步要继续推进，我建议直接按第一阶段开始，优先做：

1. `REQ-010`
2. `REQ-012`
3. `REQ-014`
4. `REQ-013`

这样可以最快形成“资产可定位 + 输入可读 + 时间可推进”的稳定底座。
