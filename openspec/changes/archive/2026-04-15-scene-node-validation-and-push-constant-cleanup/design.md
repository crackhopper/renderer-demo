## Context

当前对象渲染路径由 `RenderableSubMesh` 直接向 `RenderQueue` 暴露 vertex/index/descriptors/objectPC/shader 等底层资源，`RenderQueue::buildFromScene(...)` 在组装 `RenderingItem` 时顺便决定对象是否“够资格”进入某个 pass。与此同时，`PC_Draw` 仍携带 `enableLighting`、`enableSkinning` 这类会改变 shader 接口与 pipeline identity 的开关，导致：

- push constant ABI 和 shader variant 责任混杂；
- mesh/material/skeleton 的非法组合只能在较晚阶段暴露；
- `Skeleton` 继续参与 object render signature，使 pipeline identity 仍受对象侧资源存在性影响；
- `IRenderable` 名称与实际职责不匹配，难以承载更高层的 scene object 语义。

本变更跨越 `core/scene`、`core/frame_graph`、`core/rhi`、`infra/shader_compiler` 与材质 loader，是一次显式的对象模型与验证责任重构。

## Goals / Non-Goals

**Goals:**

- 稳定引擎级 push constant ABI，本期统一为仅含 `model` 的结构。
- 把 `lighting/skinning` 这类结构性差异提升到 `MaterialTemplate` / shader variant。
- 引入高层 `SceneNode`，让结构性校验在构造与结构性 setter 时立即发生。
- 让 `supportsPass(pass)` 和 queue 组装都建立在 pass 级 validated-entry 缓存上。
- 移除 `Skeleton` 对 pipeline identity 的直接贡献，只保留运行时资源职责。
- 把大部分测试控制在 `core` / `infra`，降低 backend 依赖。

**Non-Goals:**

- 本期不引入树状 scene graph、parent/child transform hierarchy、节点局部/世界变换传播。
- 本期不设计多版本 push constant ABI 管理机制。
- 本期不把 `objectPC` 从 `SceneNode` 移除，只将其语义收敛为统一 `model` 的承载体。
- 本期不把所有运行时 descriptor 参数缓存进 validated-entry；缓存只覆盖结构性结果。

## Decisions

### Decision: 统一 push constant ABI 为 `PC_Base`

渲染主路径统一采用只包含 `Mat4f model` 的 engine-wide push constant 结构。`PC_Draw` 可以保留为别名或过渡扩展点，但不得继续承载会改变 shader 接口或 pipeline identity 的字段。

这样做的原因：

- 先稳定 ABI，再演进 variant 系统，比同时支持多套 ABI 更可控。
- `enableLighting` / `enableSkinning` 会改变 shader 代码路径、vertex 输入契约与 descriptor 需求，不应停留在 draw-time push constant。

备选方案：

- 继续沿用 `PC_Draw` 并在 queue/backend 层解释 feature 开关。
  该方案延续了“非法组合晚暴露”的问题，拒绝采用。
- 直接引入多版本 push constant registry。
  当前真实用例不足，复杂度过高，暂不采用。

### Decision: 用 `SceneNode` 替换当前高层 renderable 语义

新增高层 `IRenderable` 语义和其主要实现 `SceneNode`。`SceneNode` 负责持有 `nodeName`、`mesh`、`materialInstance`、可选 `skeleton` 与过渡态 `objectPC`，并在构造与结构性 setter 中完成结构性自校验。

这样做的原因：

- `SceneNode` 可以脱离 `Scene` 独立存在，非法装配在对象创建时立即失败。
- `Scene` 成为命名空间和容器，`RenderQueue` 成为已验证结果的消费者，职责边界更稳定。

备选方案：

- 保持 `RenderableSubMesh`，只在 `RenderQueue::buildFromScene(...)` 增加更多检查。
  这会继续让 queue 承担对象建模错误的发现职责，不采用。

### Decision: 维护 `pass -> validated entry` 结构性缓存

`SceneNode` 为每个当前启用且通过校验的 pass 缓存一个 validated entry。该 entry 只保存 queue 构造 `RenderingItem` 所需的稳定结构性结果，例如：

- 已验证的 shader/material pass 结构；
- mesh vertex/index 资源引用；
- 结构性 descriptor 资源集合；
- object render signature 组成结果；
- 用于快速回答 `supportsPass(pass)` 的缓存状态。

`supportsPass(pass)` 只读取 material pass enable 状态和缓存，不在调用时重跑推导。`RenderQueue` 只从 validated entry 组装 item，并追加 scene-level resources。

这样做的原因：

- 避免重复做 variant / vertex layout / descriptor legality 推导。
- 明确区分“结构性缓存”和“运行时参数”，避免把 UBO 值、纹理内容等易变数据错误缓存进节点。

备选方案：

- 只缓存 `bool`。
  无法承载 queue 组装稳定所需信息，不采用。
- 不缓存，按需每次重新验证。
  增加重复工作，也让 `supportsPass` 含有隐式副作用，不采用。

### Decision: `Skeleton` 从 pipeline identity contributor 降级为 legality dependency

pipeline 是否为 skinned 只由 material-side shader variants 决定；`Skeleton` 不再贡献 object render signature，也不再直接影响 `PipelineKey`。但当 variant 需要 skinning 时，`SceneNode` 必须提供 `Skeleton` / Bones UBO；当 variant 不需要 skinning 时，存在 `Skeleton` 也不得切出新 key。

这样做的原因：

- pipeline 形态应由 shader/program 形态决定，而不是由对象是否“恰好挂了一个 skeleton”决定。
- `Skeleton` 是 per-object runtime resource，不适合与材质模板身份耦合。

备选方案：

- 继续把 `Skeleton` 组合进 object render signature。
  会维持错误的 identity 维度，不采用。

### Decision: 把 vertex 输入反射纳入合法性校验链

扩展 `ShaderReflector` 以反射 vertex stage 输入 attribute，并在 `SceneNode` 结构性校验中统一比对：

- shader 当前 variant 需要的 vertex inputs；
- mesh `VertexLayout` 实际提供的 inputs；
- shader descriptor 需求与 material/skeleton 可提供资源；
- skinning variant 与 skeleton/Bones 资源存在性。

所有结构性失败统一按程序员错误处理，记录 `FATAL` 日志并终止进程。

这样做的原因：

- `USE_SKINNING` 改变的不只是 shader 分支，还会改变 vertex 输入契约。
- 只有把反射和对象装配放到同一条校验链里，才能在 queue 之前发现错误。

备选方案：

- 仅检查 descriptor，不检查 vertex inputs。
  无法覆盖 variant 改变 vertex ABI 的核心风险，不采用。

## Risks / Trade-offs

- [Risk] `SceneNode` 缓存与 `MaterialInstance` pass enable 状态不同步。 → Mitigation: 明确要求 pass enable 变化经 `Scene` 或节点回调触发缓存失效与立即重建。
- [Risk] 结构性校验统一 `FATAL + terminate` 会让测试失败表现更“硬”。 → Mitigation: 把大部分覆盖放在 core/infra 层，使用专门的死亡测试或子进程测试策略。
- [Risk] `IRenderable` 语义切换会影响现有调用点与测试夹具。 → Mitigation: 在 specs 和任务里要求先完成新对象模型，再迁移 queue/scene 调用点。
- [Risk] vertex input 反射细节不足，可能遇到未来更复杂的输入形态。 → Mitigation: 本期只覆盖当前 forward path 所需的 top-level attribute 契约，复杂输入留待后续扩展。

## Migration Plan

1. 先收敛 push constant 结构与 shader/material variant 归属。
2. 扩展 shader reflection，补齐 vertex input 元数据。
3. 引入 `SceneNode` 与 validated-entry 缓存，在 core 层完成自校验路径。
4. 迁移 `Scene` 与 `RenderQueue` 到新对象模型，并移除 queue 内的结构性判定。
5. 更新 render-signature / skeleton identity 行为与测试。

回滚策略：

- 若中途发现 variant/validation 路径不完整，可临时保留旧 `RenderableSubMesh` 适配层，但不得恢复 skeleton 参与 pipeline identity。

## Open Questions

- `MaterialInstance` 的 pass enable 状态变化通过何种通知机制传递给 `SceneNode`，是节点直接订阅，还是先由 `Scene` 中转。
- validated entry 中是否需要保留显式 debug payload，方便 `FATAL` 日志直接打印最近一次成功校验的结构摘要。
- 当前是否要把旧 `RenderableSubMesh` 直接移除，还是短期保留一个兼容包装类型以降低迁移摩擦。
