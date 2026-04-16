# 什么是 Pipeline

在材质系统里，`Pipeline` 这个词很容易显得又大又抽象。更容易理解的方式是：

它回答的是两个非常具体的问题：

1. 两个 draw 能不能复用同一套 GPU 渲染配置？
2. 如果不能复用，引擎该拿哪些结构信息去新建一套配置？

所以这里说的 pipeline，不只是 Vulkan 最后那个对象句柄，也不是单独某个 shader。它更像是“把几何、材质、pass 和 render state 收拢成一套可复用渲染配置”的那条链路。

## 为什么在材质系统里要先理解它

因为材质系统并不只是“保存几个颜色值和贴图”。

材质系统真正决定的东西里，有一部分会直接改变 draw 的结构：

- 用哪组 shader
- 开了哪些 variants
- 这个 pass 的 render state 是什么
- 这个对象参加哪些 pass

这些都不是“改个参数值”那么简单。一旦它们变化，引擎很可能就不能继续复用原来的 pipeline。

也正因为这样，理解材质系统时，早点把 pipeline 的角色看清楚，会更自然。后面再看“模板为什么影响 pipeline”，就不会觉得它是一个突兀的话题。

## 我们到底在复用什么

可以先把 pipeline 想成“一套已经确定好的渲染结构”。

这套结构至少包括：

- vertex input 长什么样
- primitive topology 是什么
- shader stages 是什么
- descriptor bindings 是什么
- render state 是什么

如果两次 draw 在这些结构性条件上完全一样，那么它们就有机会复用同一条 pipeline。

如果其中任何关键条件不同，比如 shader variant 不同、render state 不同、顶点布局不同，那么通常就要换一条 pipeline。

## 一条 pipeline 是怎样被确定下来的

在当前项目里，这条路径可以粗略理解成四步：

1. 几何提供 mesh layout、topology 这类对象侧结构
2. 材质模板提供 pass、shader、variants 和 render state 这类材质侧结构
3. `SceneNode` 在某个 pass 下把对象和材质拼成一份 validated 数据
4. 这份数据继续整理成 `PipelineKey` 和 `PipelineBuildDesc`

这里可以把两个名字先记住：

- `PipelineKey`：回答“是不是同一条 pipeline”
- `PipelineBuildDesc`：回答“如果要建，需要哪些输入”

一个偏身份，一个偏构建。两者相关，但不是同一个东西。

## 什么通常不会切换 pipeline

不是所有材质变化都会切 pipeline。

通常不会直接切 pipeline 的，是这些更像“运行时数据”的东西：

- `setFloat`、`setVec3`、`setVec4`
- 纹理内容变化
- camera / light 这类 scene-level 资源里的值变化

这些变化当然会影响最终画面，但它们通常不会改变 draw 的结构本身。

这也是为什么当前项目要把材质拆成 `MaterialTemplate` 和 `MaterialInstance`：

- template 负责结构
- instance 负责运行时值

## 和材质系统最直接的关系是什么

如果只从材质系统的角度记一件事，那就是：

> 不是“材质会不会参与 pipeline”，而是“材质里哪些部分会参与 pipeline”。

当前会稳定影响 pipeline 的，主要是模板这一层的内容：

- pass 定义
- shader name
- enabled variants
- render state

而 `MaterialInstance` 里的普通参数写入，通常不会改变 pipeline identity。

所以先把 pipeline 理解成“渲染结构的复用单位”，后面再看模板如何影响它，就会顺很多。

## 继续往下读

- 接着看 [模板如何影响 Pipeline](template-and-pipeline.md)
- 想看更底层的实现整理：[`../../subsystems/pipeline-identity.md`](../../subsystems/pipeline-identity.md)
- 想看缓存和构建复用：[`../../subsystems/pipeline-cache.md`](../../subsystems/pipeline-cache.md)
