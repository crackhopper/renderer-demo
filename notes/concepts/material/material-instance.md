# 材质实例如何承接运行时状态

这篇文档只讨论 `MaterialInstance`。如果 `MaterialTemplate` 是蓝图，那么 `MaterialInstance` 就是当前 scene 真正会拿在手里的材质对象。

## 为什么 instance 是材质系统的运行时中心

当前项目里，scene、render queue 和 backend 真正持有的材质类型，已经统一成 `MaterialInstance`。

这意味着它要同时承接几类运行时责任：

- 材质参数
- 材质自己的 descriptor 资源
- instance 级 pass enable 状态
- 和 scene 结构性变化有关的通知

所以 `MaterialInstance` 不是一个薄壳。它是当前材质系统里最像“运行时对象”的那一层。

## 它现在保存了哪些状态

从类定义上看，`MaterialInstance` 当前主要持有：

- `m_template`
- `m_uboBuffer`
- `m_uboBinding`
- `m_uboResource`
- `m_textures`
- `m_enabledPasses`
- `m_passStateListeners`

这几块状态刚好对应几类职责：

- template 来源
- 反射驱动的材质 UBO
- 纹理资源
- enabled pass 子集
- 与 scene 的结构性传播回调

## `MaterialUBO` 是怎样被构造出来的

instance 构造时，会从 enabled pass shader 的反射结果里找 `MaterialUBO`。

这里有两个很重要的行为：

- 它优先看 enabled passes 对应的 shader
- 如果多个 enabled pass 都有 `MaterialUBO`，它们的布局必须一致

只有布局一致，instance 才会分配一份统一的 `m_uboBuffer` 和 `MaterialParameterDataResource`。如果找不到，再回退去看 template-level shader。

这说明当前实现默认接受这样的前提：

一个材质实例虽然可以跨多个 pass 运行，但这些 enabled pass 对材质 UBO 的理解必须一致。

## 运行时参数是怎样写进去的

当前 `setVec4`、`setVec3`、`setFloat`、`setInt` 最终都会走同一条路径：

- 根据名字在 `m_uboBinding->members` 里查成员
- 验证类型是否匹配
- 按反射 offset 把值 `memcpy` 到 `m_uboBuffer`

因此，这些 setter 不是手写 offset 的包装，而是完全依赖反射结果来定位材质参数。

`setTexture` 也是类似思路，只不过它现在走的是 template 级的按名查找缓存，再把资源存进 `m_textures`。

这里要注意一个当前限制：这个查找还没有带上 pass 作用域，所以它默认假设不同 pass 中同名 sampler 的位置是一致的。只要模板开始承载多套 layout 不同的 pass，这个假设就会变得不可靠。这个缺口已经记录在 [`REQ-030`](../../requirements/030-pass-scoped-material-binding-resolution.md)。 

## pass enable 为什么算结构变化

`MaterialInstance` 还有一组比普通参数更“重”的状态：

- `isPassEnabled(pass)`
- `setPassEnabled(pass, enabled)`
- `getEnabledPasses()`

这些状态之所以重要，是因为它们不只是影响 draw 值，而是会影响：

- 这个对象参加哪些 pass
- `SceneNode` 需要校验哪些 pass
- `getPassFlag()` 和 descriptor 资源在 pass 维度上的视图

也因此，当前实现把 pass enable 变化通过 listener 通知给 scene，由 scene 去触发引用该材质实例的节点重验证。

普通的 `setFloat` / `setTexture` / `syncGpuData()` 则不走这条传播链。

## descriptor 资源是怎样整理出来的

`MaterialInstance::getDescriptorResources()` 当前会返回一组确定顺序的资源：

- 先是材质自己的 `MaterialUBO`
- 再是按 `(set << 16 | binding)` 排序后的纹理资源

这个排序很重要，因为它让 descriptor 列表在运行时保持稳定，不需要依赖 map 的遍历顺序。

## 往实现层再走一步

如果要继续跟代码，最值得看的就是：

- [material_instance.hpp](/home/lx/proj/renderer-demo/src/core/asset/material_instance.hpp:38)
- [material_instance.cpp](/home/lx/proj/renderer-demo/src/core/asset/material_instance.cpp:40)
- [`../../subsystems/material-system.md`](../../subsystems/material-system.md)
- `openspec/specs/material-system/spec.md`
