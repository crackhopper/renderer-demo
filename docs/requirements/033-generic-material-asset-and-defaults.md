# REQ-033: 通用材质资产与默认值合同

## 背景

仅靠 shader reflection 还不足以支撑一个真正通用的材质 authoring 流程。

reflection 能稳定告诉系统：

- 有哪些 pass
- 有哪些 binding
- 每个 binding 的 descriptor 类型
- buffer 成员布局与 offset

但 reflection 不能回答这些更高层的问题：

- 这个材质实例的默认参数值是什么
- 默认纹理/立方体贴图资源是什么
- 哪些参数应该暴露给工具/UI
- 哪些参数属于全局默认，哪些参数只在某个 pass 下覆写
- 新增一个材质模板时，如何避免每次都单独写一个定制 loader

如果继续把这些语义散落在代码里的特化 loader 中，材质系统虽然能运行，但无法形成通用资产路径。

因此，在 [`REQ-031`](031-global-shader-binding-contract.md) 和 [`REQ-032`](032-pass-aware-material-binding-interface.md) 定义了 ownership 与 runtime interface 之后，还需要一个外部 material asset 合同，用来描述默认值和 authoring metadata。

## 目标

1. 为材质提供一个通用的外部资产格式，避免每新增模板都开发专用 loader。
2. 让默认参数值、默认资源引用、参数显示分组等信息脱离代码硬编码。
3. 保持 ownership 仍由 shader contract + reflection 决定，而不是被外部配置重写。
4. 为后续编辑器和资产管线提供稳定的 material authoring 入口。

## 需求

### R1: 首版必须定义统一的 material asset 文件格式

首版必须定义一个统一的 `yaml` 材质资产格式。

该格式至少要能表达：

- 所属 shader / material template
- 全局默认参数值
- 全局默认资源引用
- pass 级默认值覆写
- 参数的 authoring metadata，例如显示名、分组、是否暴露给工具

首版不得再把“新增一个可配置材质模板”作为必须开发新 loader 的前提。

### R2: Material asset 文件不得参与 ownership 判定

material asset 文件可以描述：

- 默认值
- 默认资源
- authoring metadata
- pass 级覆写

但不得描述或覆写：

- 哪些 binding 属于 material-owned
- 哪些 binding 属于 system-owned

ownership 的唯一规则来源仍然是：

- [`REQ-031`](031-global-shader-binding-contract.md) 中的保留名字集
- shader reflection 的实际 binding 集

### R3: 参数与资源合法性必须由 shader reflection 决定

material asset 文件中出现的：

- parameter binding 名
- member 名
- texture / cube / buffer binding 名
- pass 名

都必须能在 shader reflection 与 material interface 中找到对应项。

也就是说：

- `yaml` 可以补充默认值与 metadata
- 但不能把不存在的 binding/member 通过配置“声明出来”

### R4: 首版必须支持全局默认值与 pass 级覆写

material asset 文件必须支持两层默认值：

- 全局默认值
- `passes.<pass>` 下的局部覆写

推荐合同：

- 不带 pass 的默认值用于所有 pass
- `passes.<pass>.parameters` 与 `passes.<pass>.resources` 可对局部值覆写

这与 `REQ-032` 中“同名 binding 可跨 pass 共存，但运行时解析必须显式带 pass”的合同保持一致。

### R5: 参数写入模型必须与 runtime API 对齐

material asset 文件中的 buffer 参数必须按：

- `bindingName + memberName`

表达，而不是只按 member 名。

它必须与 `REQ-032` 的 runtime API 方向对齐，例如：

- `setParameter(bindingName, memberName, value)`
- `setParameter(pass, bindingName, memberName, value)`

首版不得定义与 runtime API 语义不一致的另一套参数命名规则。

### R6: 首版资源默认值只支持简单引用模型

首版 material asset 文件对资源默认值只要求支持：

- 资源路径
- 内置占位符名字，例如 `white`、`black`、`normal`

首版不要求支持：

- sampler 状态内联定义
- 复杂 import graph
- 运行时表达式

这样可以先把通用 loader 打通，再逐步扩展资源 authoring 能力。

### R7: `yaml` 中的参数列举不是白名单约束

material asset 文件可以列出：

- 默认值
- 暴露给 UI 的参数
- authoring metadata

但它不是 shader 参数合法性的白名单来源。

如果某个 binding/member 在 shader reflection 中存在，而 `yaml` 没列出来：

- 它仍然是合法的 material-owned slot
- 只是在当前 asset 中没有默认值或 authoring metadata

### R8: 通用 loader 必须成为首版正式路径

首版必须允许一个通用 material loader 完成至少以下流程：

1. 读取 `yaml` 材质资产
2. 加载 shader / template
3. 读取 reflection
4. 根据 `REQ-031` / `REQ-032` 构建 material interface
5. 应用默认参数与默认资源
6. 生成 `MaterialTemplate` / `MaterialInstance`

对于普通材质模板，系统不得再要求每种材质都编写专门的硬编码 loader。

## 测试

- 一个带 `yaml` 的材质资产可以在不编写专用 loader 的情况下实例化出可用材质
- `yaml` 中声明的参数与资源名字若不在 reflection 中存在，系统必须明确报错
- 全局默认值能作用到所有 pass，`passes.<pass>` 覆写能只影响目标 pass
- `yaml` 中未列出的合法 shader 参数不会因此失效
- `yaml` 若试图声明 ownership 或覆写 system-owned/material-owned 归属，系统必须拒绝
- 默认资源可通过资源路径或内置占位符成功解析

## 修改范围

- 通用 material asset schema 文档
- 材质 loader 路径
- `src/core/asset/material_template.*`
- `src/core/asset/material_instance.*`
- `notes/subsystems/material-system.md`
- `notes/subsystems/shader-system.md`
- `openspec/specs/material-system/spec.md`
- 资产管线相关说明文档

## 边界与约束

- 本 REQ 不改变 ownership 规则；ownership 由 [`REQ-031`](031-global-shader-binding-contract.md) 定义
- 本 REQ 不重新定义 runtime material interface；runtime 合同由 [`REQ-032`](032-pass-aware-material-binding-interface.md) 定义
- 首版只要求 `yaml`，不要求 editor、graph-based material authoring、或复杂 schema 继承机制
- 首版不要求把所有资源导入管线问题一并解决

## 依赖

- [`REQ-031`](031-global-shader-binding-contract.md)
- [`REQ-032`](032-pass-aware-material-binding-interface.md)
- `openspec/specs/material-system/spec.md`

## 后续工作

- 如果后续需要 editor-facing schema、参数分组规范或 schema versioning，可继续追加 requirement
- 如果后续需要更复杂的 sampler/state authoring，可单独扩展资源 authoring 合同
- 如果后续需要真正的 material graph authoring，应在本 REQ 之上另立更高层资产 requirement

## 实施状态

未开始。
