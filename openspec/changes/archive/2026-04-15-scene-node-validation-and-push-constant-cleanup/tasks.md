## 1. Push Constant And Material Variant Cleanup

- [x] 1.1 收敛 `PC_Draw` / shader push constant 定义为仅包含统一 `model` 的 ABI，并移除 lighting/skinning feature 字段
- [x] 1.2 更新 `blinnphong_0` 材质加载路径，使 variant 集合由 loader 决定并同时写入 shader 编译输入与 `RenderPassEntry::shaderSet.variants`
- [x] 1.3 调整相关 core/infra 测试，验证 runtime 参数写入不影响 template-owned variant identity

## 2. Shader Reflection Support

- [x] 2.1 扩展 `ShaderReflector` / `CompiledShader`，反射并暴露 vertex-stage input attributes
- [x] 2.2 为 variant 相关 shader 反射添加 infra 层测试，覆盖 skinned 与 non-skinned vertex input 契约差异

## 3. SceneNode Validation Model

- [x] 3.1 引入高层 `SceneNode` / `IRenderable` 结构，要求 `nodeName`、`mesh`、`materialInstance` 必填，`skeleton` 可选，并保留 `objectPC`
- [x] 3.2 实现 `SceneNode` 构造时和结构性 setter 中的统一结构性校验，失败时执行 `FATAL + terminate`
- [x] 3.3 实现 `pass -> validated entry` 缓存、`supportsPass(pass)`、以及对 `MaterialInstance` pass enable 变化的缓存失效与重建路径

## 4. Pipeline Identity And Queue Migration

- [x] 4.1 调整 object render signature 与相关类型，使 `Skeleton` 不再参与 `PipelineKey`，skinning 差异只来自 material-side variants
- [x] 4.2 迁移 `Scene` 到显式 `sceneName` + `nodeName` 唯一约束模型，并允许 `SceneNode` 脱离 `Scene` 独立存在
- [x] 4.3 重构 `RenderQueue::buildFromScene(...)` 只消费已验证的节点结果，不再做首次结构性校验

## 5. Verification

- [x] 5.1 新增 core 层测试，覆盖 SceneNode 自校验、缓存命中/失效、重复 `nodeName` 拒绝、以及 `supportsPass(pass)` 的缓存语义
- [x] 5.2 新增或更新 infra 层测试，覆盖 vertex layout 与 reflected shader input 的匹配/失败路径
- [x] 5.3 更新 pipeline identity 与 render queue 测试，验证“挂 Skeleton 不再切 key，variant 变化才切 key”，并确认 queue 仅消费已验证结果
