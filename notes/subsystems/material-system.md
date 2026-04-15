# Material System

> 材质系统的核心不是一堆独立状态，而是 `MaterialTemplate + MaterialInstance` 的分层：template 持有 pass 结构与 shader variants，instance 只持有运行期参数、资源和 pass enable 状态。
>
> 权威 spec: `openspec/specs/material-system/spec.md`

## 它解决什么问题

- 把 shader、render state、shader variants、材质参数组织成稳定的运行期对象。
- 避免手写 uniform offset 和 descriptor 绑定表。
- 让材质 pass 结构直接参与 `PipelineKey` 生成。

## 核心对象

- `MaterialTemplate`：定义某个材质有哪些 pass、每个 pass 用什么 shader、variants 和 render state。
- `MaterialInstance`：持有运行期参数，是 `IMaterial` 的唯一实现。
- `RenderPassEntry`：单个 pass 的 shader 配置和 render state。
- `UboByteBufferResource`：把材质内部的 UBO byte buffer 暴露给 backend。

## 典型数据流

1. loader 为每个 pass 决定 shader variants，并编译得到对应 `CompiledShader`。
2. `MaterialTemplate` 持有 pass entries，并把 pass shader 的反射结果并入 template 级 binding cache。
3. `MaterialInstance` 构造时从“已启用 pass 对应 shader”里选取 `MaterialUBO` 布局，分配 byte buffer。
4. 运行时通过 `setVec4` / `setVec3` / `setFloat` / `setInt` / `setTexture` 写参数。
5. `setPassEnabled(pass, enabled)` 只改变 pass 可用性，并通知 `SceneNode` 之类的监听者重建结构缓存。
6. `updateUBO()` 把 dirty 状态传给 `m_uboResource`。

## 关键约束

- shader 里的材质 UBO 名必须是 `MaterialUBO`。
- shader variants 属于 template/pass，不属于 instance；运行时改 UBO 或 texture 不会产生新的 pipeline identity。
- `MaterialInstance` 会断言所有已启用 pass 的 `MaterialUBO` 布局一致；不一致视为程序错误。
- `setTexture` 绑定的是 `CombinedTextureSampler`，不是裸 texture。
- `getDescriptorResources()` 的顺序固定：先 UBO，再按 `(set << 16 | binding)` 升序排好的纹理资源。
- 当前 engine-wide draw push constant ABI 只有 `model`，lighting / skinning 不再通过 push constant 切接口。

## 当前实现边界

- `MaterialTemplate` 仍保留 `RenderPassEntry::bindingCache` 这条旧路径，但运行时主路径主要依赖 template 级 `m_bindingCache`。
- `MaterialInstance::getRenderState()` 仍是过渡接口，默认走 `Forward` entry；真正的 pass-sensitive 身份路径已经走 `getRenderSignature(pass)` 和 `getShaderInfo(pass)`。
- 若 enabled passes 没有任何 `MaterialUBO`，实例会回退到 template shader 查找布局。
- `loadBlinnPhongMaterial()` 已经开始按 pass 保存 variants，并驱动 forward/shadow 等 pass-specific shader 选择。

## 从哪里改

- 想加新材质参数：看 shader 反射和 `MaterialInstance` setter 路径。
- 想改 variant 身份：看 `ShaderProgramSet`、loader 和 `RenderPassEntry`。
- 想改 pass enable 对 scene 的影响：看 `MaterialInstance::setPassEnabled()` 和 listener 机制。

## 关联文档

- `notes/subsystems/shader-system.md`
- `notes/subsystems/pipeline-identity.md`
- `notes/subsystems/scene.md`
