## Why

当前渲染对象装配把两类职责混在了一起：push constant 同时承载每-draw 数据和会改变 shader/pipeline 形态的静态开关，而 `RenderQueue` 又在收集阶段隐式承担对象合法性判定。这让非法 mesh/material/skeleton 组合只能在较晚阶段暴露，也让 pipeline identity 继续被对象侧 `Skeleton` 意外放大。

这个变更把结构性约束前移到场景对象装配阶段，稳定统一的 push constant ABI，并把 shader variant 的归属收敛到 `MaterialTemplate`。这样可以让 `SceneNode` 在脱离 `Scene` 的情况下独立完成自校验，`RenderQueue` 只消费已验证结果。

## What Changes

- 收敛引擎级 push constant ABI，只保留统一的 `model` 数据；移除 `enableLighting` / `enableSkinning` 这类结构性开关。
- 新增高层 `IRenderable` / `SceneNode` 抽象，`SceneNode` 必需持有 `nodeName`、`mesh`、`materialInstance`，可选持有 `skeleton`，并保留 `objectPC` 作为过渡字段。
- `SceneNode` 构造即执行结构性校验；`setMesh(...)`、`setMaterialInstance(...)`、`setSkeleton(...)` 等结构性 setter 会立即重校验并刷新 pass 级缓存。
- 引入 `pass -> validated entry` 的结构性缓存，`supportsPass(pass)` 直接基于材质 pass 启用状态和缓存结果回答，不再临时推导。
- `Skeleton` 不再参与对象 render signature / `PipelineKey` 主路径；skinning 差异仅由 material-side shader variants 决定。
- `RenderQueue` 只消费已验证的结构性结果，不再做 mesh/material/skeleton 组合合法性校验。
- `Scene` 需要显式 `sceneName`，并在同一 `Scene` 内强制 `nodeName` 唯一。
- 所有结构性校验失败统一视为程序员错误，采用 `FATAL + terminate`。
- 测试新增和变更尽量限定在 `core/` 与 `infra/`，避免把验证职责压进 backend 测试。

## Capabilities

### New Capabilities
- `scene-node-validation`: 定义高层 `SceneNode` / `Scene` 命名与结构性校验模型、pass 级 validated-entry 缓存，以及 `RenderQueue` 只消费已验证结果的契约。

### Modified Capabilities
- `frame-graph`: `RenderQueue` 从场景节点提取稳定结构性结果，不再承担首次对象合法性判定。
- `material-system`: 统一 push constant 只保留 `model`，并明确 variants 属于 `MaterialTemplate` / loader 产物，不属于 `MaterialInstance`。
- `render-signature`: 对象侧 render signature 不再组合 `Skeleton`，pipeline identity 中的 skinning 差异改由 material-side variants 体现。
- `shader-reflection`: 扩展反射结果以覆盖 vertex stage 输入契约，为 mesh vertex layout 与 shader variant 的结构性校验提供依据。
- `skeleton-resource`: `Skeleton` 从 pipeline identity contributor 降级为 runtime resource provider 与 legality dependency。

## Impact

- 影响代码主要在 `src/core/scene/`、`src/core/frame_graph/`、`src/core/rhi/`、`src/infra/shader_compiler/` 与材质 loader 路径。
- 影响现有场景对象构造方式、`IRenderable` 语义、`RenderQueue` 组装路径，以及 pipeline identity 的来源。
- 需要新增/调整 core 与 infra 层测试，覆盖 SceneNode 自校验、缓存失效与刷新、variant 驱动的 pipeline identity、以及 vertex 输入反射校验。
