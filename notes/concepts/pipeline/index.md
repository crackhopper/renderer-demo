# 一条 pipeline 是怎样被确定的

这篇文档讨论的不是 Vulkan API 细节，而是更靠前的一层：在当前项目里，pipeline 身份是什么、它为什么存在，以及 scene 里的数据怎样一路整理成 pipeline 构建输入。

前面那些概念页里，只要提到：

- `PipelineKey`
- render signature
- `PipelineBuildDesc`
- pipeline cache

都可以把这里当作总入口。

## 我们到底在说什么

在这个项目里，“渲染管线”不是单指 Vulkan 最终创建出来的 pipeline handle。

它更像一整条判断和整理链路，主要回答两个问题：

1. 两个 draw 能不能复用同一条 pipeline？
2. 如果要建一条新的 pipeline，backend 需要的完整输入到底是什么？

第一个问题，对应的是 pipeline 身份；第二个问题，对应的是 pipeline 构建输入。

## 这一层解决什么问题

这个系统的价值在于，它把“是否复用”和“如何构建”拆开了。

如果没有这层整理，backend 每次都只能拿到一堆零散状态，自己猜：

- 这次和上次是不是同一条 pipeline
- vertex input 是什么
- shader stages 是什么
- render state 是什么

现在这些事情在 core 层就已经有了更稳定的表达：

- `PipelineKey` 用来表示“是不是同一条 pipeline”
- `PipelineBuildDesc` 用来表示“如果要建，该拿哪些输入去建”

## 从 scene 走到 pipeline 的那条线

可以把这条链路理解成五步：

1. [资产系统](../assets/index.md) 里的网格对象提供几何输入
2. [材质系统](../material/index.md) 提供 shader、pass、render state 和 variants
3. [场景对象](../scene/index.md) 在 pass 级结构校验后，整理出稳定的 validated 数据
4. `RenderQueue::buildFromScene(...)` 把它们装配成 `RenderingItem`
5. `PipelineBuildDesc::fromRenderingItem(item)` 再提取 backend 真正需要的构建输入

在这个过程中，`PipelineKey` 会跟着 validated 数据一起被确定下来。

## 什么会真正影响 pipeline 身份

在当前实现里，真正稳定影响 pipeline 身份的，主要是这些结构性因素：

- mesh 的顶点布局
- primitive topology
- shader set 和 enabled variants
- render state
- pass 维度上的材质定义

反过来，有些东西通常不会直接切 pipeline：

- 普通材质参数写入
- 纹理内容变化
- camera / light 的 scene-level 资源内容变化

这些变化可能会影响 draw 结果，但不一定会改变 `PipelineKey`。

## 当前代码已经走到哪一步

这条主路径已经基本建立起来了：

- `getRenderSignature()`
- `PipelineKey`
- `ValidatedRenderablePassData`
- `PipelineBuildDesc::fromRenderingItem(...)`
- `PipelineCache`

都已经是当前代码里真实存在的部分。

所以这里不是“未来想这么做”，而是在解释“当前主路径已经怎样工作”。

## 往实现层再走一步

从实现上看，可以粗略记成两段。

第一段是“前端整理”：

- `SceneNode` 做 pass 级结构校验
- validated 数据里带上 object-side 和 material-side signature
- 组合出 `PipelineKey`

第二段是“后端消费”：

- `RenderQueue` 生成 `RenderingItem`
- `PipelineBuildDesc` 从 item 提取构建输入
- `PipelineCache` 根据 key 做预构建、查找和运行时 miss 处理

继续展开时，可以参考：

- [`../../subsystems/pipeline-identity.md`](../../subsystems/pipeline-identity.md)
- [`../../subsystems/pipeline-cache.md`](../../subsystems/pipeline-cache.md)
- [`../../subsystems/frame-graph.md`](../../subsystems/frame-graph.md)
