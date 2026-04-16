# 材质模板为什么会影响 pipeline

这篇文档只讨论一个问题：为什么 `MaterialTemplate` 这张蓝图，不只是 scene 配置的一部分，还会直接进入 pipeline 身份和构建链路。

## 先说结论

在当前项目里，template 会影响 pipeline，不是因为它“名字叫模板”，而是因为它保存了那些真正会改变渲染结构的东西：

- pass 的定义
- shader name 与 enabled variants
- render state

这些东西一旦变化，后续 scene 校验、`PipelineKey` 和 `PipelineBuildDesc` 就都可能变化。

## 从 `MaterialPassDefinition::getRenderSignature()` 开始看

当前 `MaterialPassDefinition` 的 render signature 很直接，就是把两部分组合起来：

- `shaderSet.getRenderSignature()`
- `renderState.getRenderSignature()`

然后 compose 成 `TypeTag::MaterialPassDefinition`。

这说明一个 pass 在 pipeline 身份里最核心的材质侧贡献，就是：

- 这一 pass 用的 shader 形状
- 这一 pass 用的 render state

而这两者都属于 template 蓝图，不属于 instance 的运行时值。

## `MaterialTemplate` 怎样把 pass signature 暴露出来

在 `MaterialTemplate` 上，当前有一个很关键的接口：

- `getRenderPassSignature(pass)`

它的含义非常直接：

我们可以按 pass 取出这个模板在该 pass 下的结构签名。

再往下走时，`MaterialInstance::getRenderSignature(pass)` 会基于这个 pass signature 继续 compose 成材质侧 render signature。然后 scene 再把：

- object-side signature
- material-side signature

组合进 `PipelineKey`。

所以从代码链路上看，template 不是“间接影响 pipeline”，而是直接通过 pass signature 进入材质侧身份。

## 为什么运行时参数通常不会切 pipeline

把 template 和 instance 分层的一个直接收益，就是我们把“结构变化”和“值变化”分开了。

通常会切 pipeline 的，是这些 template/pass 级因素：

- shader name
- enabled variants
- render state
- pass 定义本身

而 `MaterialInstance` 里的这些东西通常不会切 pipeline：

- `setFloat`
- `setVec3`
- `setTexture`
- `syncGpuData()`

它们会改 draw 结果，但不一定改 `PipelineKey`。

这也是为什么当前材质系统会把 variants 固定在 template/pass 上，而不是允许 instance 在运行时随便改 shader 结构。

## 和 scene 校验是怎样连起来的

`SceneNode` 在为每个 enabled pass 重建 validated 数据时，会同时取：

- mesh 的 `getRenderSignature(pass)`
- material 的 `getRenderSignature(pass)`

然后调用 `PipelineKey::build(...)`。

这意味着 template 里每个 pass 的定义，不只是给 backend 最后创建 pipeline 用的；它在 scene 前端就已经参与了：

- 这个对象在某个 pass 下是否合法
- 这个对象在某个 pass 下会不会和别的对象复用同一条 pipeline

## 往实现层再走一步

如果顺着这条链继续读，最值得看的是：

- [material_pass_definition.hpp](/home/lx/proj/renderer-demo/src/core/asset/material_pass_definition.hpp:103)
- [material_template.hpp](/home/lx/proj/renderer-demo/src/core/asset/material_template.hpp:30)
- [material_instance.cpp](/home/lx/proj/renderer-demo/src/core/asset/material_instance.cpp:203)
- [object.cpp](/home/lx/proj/renderer-demo/src/core/scene/object.cpp:289)
- [`../pipeline/index.md`](../pipeline/index.md)
