# 怎样定义自己的材质模板

这篇文档讨论的是“如果不想只用现成的 `loadBlinnPhongMaterial()`，当前代码允许我们怎样自己定义一张材质模板”。

## 先看今天已经存在的主路径

虽然项目里目前最成熟的入口仍然是 `loadBlinnPhongMaterial()`，但从代码结构上说，自定义材质模板这件事并不是未来想法，而是已经有基础路径：

1. 准备 shader 和 variants
2. 编译并反射得到 `CompiledShader`
3. 创建 `MaterialTemplate`
4. 为某个 pass 填 `MaterialPassDefinition`
5. `buildBindingCache()`
6. 基于 template 创建 `MaterialInstance`

也就是说，今天缺的不是“底层根本做不到”，而是“还没有把这条路径收敛成一个统一、面向使用者的 loader 契约”。

## `blinn_phong_material_loader.cpp` 已经给出了最小样板

当前最接近“自定义模板样板”的真实代码，其实就在 [blinn_phong_material_loader.cpp](/home/lx/proj/renderer-demo/src/infra/material_loader/blinn_phong_material_loader.cpp:76) 里。

这段代码做了几件很有代表性的事：

- 规范化 forward variants
- 校验 variant 组合是否合法
- 编译 `blinnphong_0.vert/.frag`
- 反射出 binding 和 vertex input
- 创建 `MaterialTemplate`
- 组装 `ShaderProgramSet`
- 构造 `MaterialPassDefinition`
- `tmpl->setPass(Pass_Forward, entry)`
- `tmpl->buildBindingCache()`
- 再创建 `MaterialInstance` 并写默认参数

换句话说，这个 loader 虽然只服务 `blinnphong_0`，但已经把“自定义模板”最关键的搭建过程演示出来了。

## 如果今天手工定义一张模板，最小步骤是什么

按当前实现，最小步骤大致可以写成：

```cpp
auto shader = std::make_shared<CompiledShader>(...);
auto tmpl = MaterialTemplate::create("my_material", shader);

ShaderProgramSet programSet;
programSet.shaderName = "my_material";
programSet.variants = variants;
programSet.shader = shader;

MaterialPassDefinition entry;
entry.shaderSet = programSet;
entry.renderState = RenderState{};
entry.buildCache();

tmpl->setPass(Pass_Forward, std::move(entry));
tmpl->buildBindingCache();

auto instance = MaterialInstance::create(tmpl);
```

这里最容易忽略的有两点：

- `ShaderProgramSet` 才是 pass 真正绑定的 shader 入口
- `buildBindingCache()` 不能漏，否则后续名字查找会不完整

## 当前还缺什么

虽然底层路径已经存在，但从“文档和使用者入口”的角度看，当前还差一层正式约定：

- loader 到底应该返回 template、instance，还是两者都暴露
- 自定义模板怎样成为稳定 API，而不是散落在具体 loader 里
- 什么东西属于 template 构造期，什么东西属于 instance 运行时

这个缺口已经单独挂在 [`REQ-025`](../../requirements/025-custom-material-template-and-loader.md)。

## 现在写这篇文档的意义是什么

意义不是说“自定义模板已经有漂亮 API 了”，而是把今天真实存在的能力边界讲清楚：

- 底层已经能定义自定义模板
- 最成熟的例子是 `blinn_phong_material_loader.cpp`
- 还缺一层统一、正式、面向使用者的入口约定

这能让后续把 `REQ-025` 落地时，有一个明确的起点，而不是重新发明材质系统的分层。

## 往实现层再走一步

继续展开时，可以参考：

- [blinn_phong_material_loader.cpp](/home/lx/proj/renderer-demo/src/infra/material_loader/blinn_phong_material_loader.cpp:76)
- [material_template.hpp](/home/lx/proj/renderer-demo/src/core/asset/material_template.hpp:13)
- [material_pass_definition.hpp](/home/lx/proj/renderer-demo/src/core/asset/material_pass_definition.hpp:103)
- [`../../subsystems/material-system.md`](../../subsystems/material-system.md)
- [`REQ-025`](../../requirements/025-custom-material-template-and-loader.md)
