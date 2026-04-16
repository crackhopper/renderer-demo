# 材质模板：一张蓝图如何描述多个 pass

这篇文档只讨论 `MaterialTemplate` 和 `MaterialPassDefinition`，也就是材质系统里最像“蓝图”的那一层。

## 从一张蓝图开始

当前项目里的 `MaterialTemplate` 很轻。它不负责保存运行时参数，也不直接拥有 instance 级状态。它做的事情更集中：

- 持有一个 template-level shader 指针
- 持有 `pass -> MaterialPassDefinition` 的映射
- 为材质系统建立一份按名字索引的 binding cache

也就是说，`MaterialTemplate` 回答的是这样的问题：

- 这个材质支持哪些 pass
- 每个 pass 用什么 shader / variants / render state
- 后续 instance 和 scene 该如何按名字找到 binding

对应代码入口在 [material_template.hpp](/home/lx/proj/renderer-demo/src/core/asset/material_template.hpp:13) 和 [material_pass_definition.hpp](/home/lx/proj/renderer-demo/src/core/asset/material_pass_definition.hpp:103)。

## `MaterialPassDefinition` 真正保存了什么

单个 pass 的定义现在集中在 `MaterialPassDefinition`：

- `renderState`
- `shaderSet`
- `bindingCache`

其中最重要的还是前两个：

- `renderState` 决定这个 pass 的光栅化 / 深度 / blend 语义
- `shaderSet` 决定 shader 名、enabled variants，以及最终绑定的 `IShaderPtr`

`bindingCache` 不是多余字段。它保存的是“这个 pass 自己那套 shader 反射绑定”，因此当一个模板有多个 pass 时，逻辑上就会有多套彼此独立的 binding 集合。

当前代码里，`MaterialTemplate::buildBindingCache()` 还会把 template shader 和各个 pass shader 的反射结果压平成一张 template-level `m_bindingCache`，供 `MaterialInstance::setTexture()` 这类按名字查找的路径复用。这个做法在“所有 pass 的同名 binding 都指向同一套布局”时还能工作，但一旦不同 pass 出现同名 binding 却位于不同 set / binding，它就会丢失 pass 作用域。这个限制已经单独记录在 [`REQ-030`](../../requirements/030-pass-scoped-material-binding-resolution.md)。 

## 为什么 template 是“多 pass 蓝图”

从接口上看，`MaterialTemplate` 最关键的动作只有几个：

- `setPass(pass, definition)`
- `getEntry(pass)`
- `getPasses()`
- `getRenderPassSignature(pass)`
- `buildBindingCache()`

这几个入口放在一起，刚好表达了一张蓝图最核心的能力：

- 我们可以往模板里注册多个 pass
- 也可以按 pass 取出对应定义
- 还可以直接从某个 pass 导出 render signature

所以在这个项目里，模板不是“材质的默认参数对象”，而是“材质有哪些 pass、这些 pass 长什么样”的集中定义。

## 这一层和 instance 的边界

`MaterialTemplate` 不负责：

- 运行时 UBO 值
- 纹理资源实例
- pass enable 子集
- shared material 的 scene 传播

这些都属于 `MaterialInstance`。

因此一个很实用的理解方式是：

- template 决定能力上限
- instance 决定运行时实际状态

这条边界也是当前材质系统能够支撑共享 instance、scene 重验证和 pipeline 身份分离的基础。

## 当前实现里还有一个细节

`MaterialTemplate::buildBindingCache()` 会把两类 shader 的反射 binding 都并进同一份名字缓存：

- template 构造时传入的 `m_shader`
- 每个 pass 的 `definition.shaderSet.getShader()`

这说明当前实现默认接受这样一种用法：

模板本身可以有一个“总体 shader 入口”，但真正每个 pass 使用的 shader 仍然以 `MaterialPassDefinition` 为准。

## 继续往下读

- 蓝图里的 shader 到底是什么：[`shader.md`](shader.md)
- 蓝图为什么会直接影响 pipeline：[`template-and-pipeline.md`](template-and-pipeline.md)
- 更偏实现视角的梳理：[`../../subsystems/material-system.md`](../../subsystems/material-system.md)
