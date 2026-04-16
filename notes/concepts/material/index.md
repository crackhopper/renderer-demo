# 材质系统总览

材质系统在这个项目里不是一个单独类，而是一条从 shader、pass、render state 到运行时参数与纹理资源的完整链路。

如果只保留一句话来理解它，那么就是：

> 我们先用 `MaterialTemplate` 定义“一个材质有哪些 pass、每个 pass 长什么样”，再用 `MaterialInstance` 承接运行时参数、纹理和 pass 开关，最后把这些结果交给 scene 与 pipeline 链路消费。

## 先建立一个整体图景

当前项目里的材质系统主要由 4 个核心部件组成：

- `MaterialTemplate`
- `MaterialPassDefinition`
- `MaterialInstance`
- `ShaderProgramSet`

它们各自分工不同：

- `MaterialTemplate` 负责蓝图
- `MaterialPassDefinition` 负责单个 pass 的 shader / variants / render state
- `MaterialInstance` 负责运行时参数、纹理和 pass enable 状态
- `ShaderProgramSet` 负责把 shader 名、variants 和编译后的 shader 绑定到一起

## 这套系统为什么重要

几何只能告诉引擎“这个对象长什么样”，还不能回答“它应该怎样被渲染”。

材质系统补上的就是这一层：

- 用哪组 shader
- 跑哪些 pass
- 用什么 render state
- 提供哪些材质参数和纹理
- 哪些变化只改值，哪些变化会改结构

也正因为这一层存在，scene 才能在前端完成结构校验，pipeline 链路也才能拿到稳定的身份和构建输入。

## 建议的阅读顺序

如果准备把这一组文档展开来读，按下面顺序最顺：

1. [材质模板：一张蓝图如何描述多个 pass](template-blueprint.md)
2. [Shader 在材质系统里扮演什么角色](shader.md)
3. [材质模板为什么会影响 pipeline](template-and-pipeline.md)
4. [材质实例如何承接运行时状态](material-instance.md)
5. [怎样定义自己的材质模板](custom-template.md)

## 当前代码已经走到哪一步

当前实现已经不是概念草图，而是主路径的一部分：

- `MaterialTemplate`
- `MaterialPassDefinition`
- `MaterialInstance`
- instance 级 pass enable / disable
- 反射驱动的 `MaterialUBO` 写入
- 按 pass 的 `RenderState` / render signature / `PipelineKey` 链路

这些都已经在当前代码里工作。

对应的权威约束主要在：

- [`../../subsystems/material-system.md`](../../subsystems/material-system.md)
- `openspec/specs/material-system/spec.md`
- [`../pipeline/index.md`](../pipeline/index.md)
