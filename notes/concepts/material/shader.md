# Shader 在材质系统里扮演什么角色

这篇文档只讨论一个问题：在当前项目里，shader 为什么不是一个孤立资源，而是材质系统的一部分。

## 先看当前代码里的对象关系

材质系统会同时接触到三层 shader 相关对象：

- `IShader`
- `CompiledShader`
- `ShaderProgramSet`

它们的角色不同：

- `IShader` 是抽象接口，暴露 stages、reflection bindings、vertex inputs、binding 查询等能力
- `CompiledShader` 是当前 `infra` 层的具体实现，承接编译和反射结果
- `ShaderProgramSet` 是材质系统真正拿来参与 pass 定义和 render signature 的值对象

所以，从材质系统视角看，shader 不是“只有一个编译结果文件”，而是“名字 + variants + 编译结果 + 反射信息”的组合。

## 为什么材质系统必须关心 shader

因为材质系统真正要决定的，不只是“选一个 shader 文件”，而是：

- 这个 pass 用哪组 shader
- 开了哪些 variants
- 反射出来有哪些 binding 和 vertex input 要求
- 这些要求是否会影响 scene 校验和 pipeline 身份

如果 shader 只是一个外部资源，而不进入材质系统，我们就很难在 pass 维度上把这些约束收拢起来。

## `ShaderProgramSet` 在这里很关键

当前项目里，`ShaderProgramSet` 保存三件事：

- `shaderName`
- `variants`
- `shader`

它一头连着 loader 生成的编译结果，一头连着 `MaterialPassDefinition` 和 render signature。

因此，材质系统里真正和 pass 绑定的，不是“单个 `IShaderPtr`”，而是 `ShaderProgramSet`。

这样做的价值很明显：

- shader 名字会进入 render signature
- enabled variants 也会进入 render signature
- 最终用于 scene 校验和 pipeline 身份的，不只是 shader 字节码，而是整个“程序集合”的形状

## 反射为什么在这一层这么重要

当前 `IShader` 会暴露两类对材质系统特别重要的反射信息：

- `getReflectionBindings()`
- `getVertexInputs()`

这两类信息会分别流向两条链：

- binding 反射决定 `MaterialUBO`、纹理 sampler、scene-level 资源如何按名字对齐
- vertex input 反射决定 `SceneNode` 如何校验 mesh 和 shader 的结构匹配

也就是说，shader 在这里不是只负责“给 GPU 跑代码”，而是还承担了一份结构合同。

## `MaterialUBO` 这个名字为什么是硬约定

当前 `MaterialInstance` 在构造时，会遍历 shader 反射 binding，专门找名字等于 `MaterialUBO` 的 uniform block。

这意味着：

- 材质系统并不是通过额外配置告诉 instance “哪个 block 是材质自己的 UBO”
- 它依赖的是 shader 里那个 block 的名字约定

这条规则已经写进当前实现和 spec 里，所以在这个项目里，`MaterialUBO` 不是随手起的名字，而是一条材质系统合同。

## 往实现层再走一步

当前这条 shader 路径大致是：

1. loader 决定 shader 名和 variants
2. `ShaderCompiler` 编译出 stages
3. `ShaderReflector` 反射出 binding 和 vertex input
4. `CompiledShader` 承接这些结果
5. `ShaderProgramSet` 把 shader 名、variants 和 `CompiledShader` 绑到一起
6. `MaterialPassDefinition` 再把它作为某个 pass 的 shader 配置保存下来

继续展开时，可以参考：

- [shader.hpp](/home/lx/proj/renderer-demo/src/core/asset/shader.hpp:118)
- [`../../subsystems/shader-system.md`](../../subsystems/shader-system.md)
- [`../../subsystems/material-system.md`](../../subsystems/material-system.md)
