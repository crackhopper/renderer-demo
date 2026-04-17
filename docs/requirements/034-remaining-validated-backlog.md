# REQ-034: 核查后剩余需求统一收敛

## 背景

2026-04-17 对 `docs/requirements/` 当前活跃需求逐项核查后，`REQ-010` 到 `REQ-018` 已具备实现与验证闭环，并已进入归档阶段。

剩余工作分成两类：

- 已有代码实现，但还差人工验收尾项
- 仍未实现或只完成前置的一组功能需求

为了避免活跃需求继续分散在多个旧文档中，本 REQ 作为当前唯一的未完成需求入口，统一承接本轮核查后确认仍未闭环的内容。原始 REQ 文档保留作为背景与设计来源，但执行应以本文件为准。

## 目标

1. 收敛当前所有未完成需求到一个统一入口
2. 明确哪些项是“实现已到位，仅缺人工验收”
3. 明确哪些项仍是实际编码工作
4. 为下一轮实现提供稳定的优先级与依赖顺序

## 需求

### R1: 完成 `REQ-019` 的人工验收尾项

`demo_scene_viewer` 的代码、构建与 headless fail-fast 已验证，但以下显示环境下的人工验收仍未完成：

- 窗口可以正常打开
- `DamagedHelmet` 与地面可见
- Stats / Camera / Directional Light / Help 四块面板可见
- `F1` 可切换 Help 面板
- `F2` 可在 Orbit / FreeFly 间切换
- Camera / Light 面板编辑能在下一帧产生可见变化
- 正常关闭时无崩溃

完成这些人工验收后，`REQ-019` 方可归档。

### R2: 删除 `RenderableSubMesh` legacy 路径

承接 `REQ-024` 的剩余工作：

- 删除 `RenderableSubMesh` 类型与相关 legacy helper
- 删除 `buildLegacyValidatedData(...)`
- 将剩余测试、示例与文档迁移到 `SceneNode`
- 回归验证 scene -> queue -> rendering item 主路径

### R3: 补齐自定义材质模板真实示例

承接 `REQ-025` 的剩余工作：

- 增加一个仓库内真实存在、不是 `blinnphong_0` 的自定义材质模板示例
- 示例需覆盖 `.material` loader 的真实使用路径
- 概念文档与 subsystem 文档需引用该示例，而不是仅停留在概念说明

### R4: 实现 Camera visibility layer / culling mask

承接 `REQ-026`：

- renderable 持有 layer mask
- camera 持有 culling mask
- queue 构建按 mask 过滤
- scene-level camera 资源收集与可见层过滤保持分离

### R5: 引入 `SpotLight`

承接 `REQ-027`：

- 增加 `SpotLight` 运行时对象
- 增加 `SpotLightUBO`
- `Scene` 与 scene-level 资源路径正式接入聚光灯
- shader / demo / 文档形成最小消费闭环

### R6: 引入 IBL scene-level 资源

承接 `REQ-028`：

- 定义最小 IBL 运行时资源集合
- 在 scene-level 资源路径挂接环境光资源
- 明确资产入口与运行时入口的边界
- 文档明确 IBL 不属于 `LightBase`

### R7: 定义统一 multi-light GPU 合同

承接 `REQ-029`：

- 将多个 light object 投影到统一 light set
- 首版采用固定上限
- 明确裁剪/排序规则
- 保证多光源数量变化不直接进入 pipeline 身份

### R8: 落地全局 shader binding ownership 合同

承接 `REQ-031`：

- 正式保留 `CameraUBO`、`LightUBO`、`Bones`
- 非保留 binding 默认归材质所有
- 保留名字误用必须 fail fast
- ownership 规则不得由外部 material asset 覆写

### R9: 落地 pass-aware material binding interface

承接 `REQ-032`：

- `MaterialTemplate` 构建 pass-aware material interface
- `MaterialInstance` 改为按 `bindingName + memberName` 写入
- descriptor 收集按 pass 解析
- 支持多个 material-owned buffer slot
- 为首版支持和不支持的 descriptor 类型给出正式行为

### R10: 定义通用 material asset 与默认值合同

承接 `REQ-033`：

- 定义统一 `yaml` 材质资产格式
- 全局默认值与 pass 级覆写
- 参数与资源合法性由 shader reflection 决定
- 通用 loader 成为正式路径

## 建议顺序

1. `R1` 完成 `REQ-019` 人工验收并归档
2. `R2` 清理 `RenderableSubMesh` legacy 债务
3. `R3` 补齐自定义材质模板真实示例
4. `R4` / `R5` / `R6` / `R7` 处理可见性与光照扩展
5. `R8` / `R9` / `R10` 处理材质系统合同升级

## 来源映射

| 来源 REQ | 当前剩余内容 |
|---|---|
| `REQ-019` | 仅剩显示环境下人工验收 |
| `REQ-024` | 删除 legacy `RenderableSubMesh` 路径 |
| `REQ-025` | 补真实自定义材质模板示例 |
| `REQ-026` | 全量未开始 |
| `REQ-027` | 前置已具备，主体未开始 |
| `REQ-028` | 未开始 |
| `REQ-029` | 仅前置完成，统一 GPU light set 未开始 |
| `REQ-031` | 未开始 |
| `REQ-032` | 未开始 |
| `REQ-033` | 未开始 |

## 依赖

- `REQ-019` 剩余人工验收依赖有显示输出的运行环境
- `R7` 依赖 `R5`
- `R9` 依赖 `R8`
- `R10` 依赖 `R8` 与 `R9`

## 实施状态

2026-04-17 新建。

- 本 REQ 由本轮需求核查结果整理而成
- 当前作为 `docs/requirements/` 的唯一活跃执行入口
