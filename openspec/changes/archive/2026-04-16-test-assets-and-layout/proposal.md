## Why

Phase 1 渲染深度（PBR / shadow / IBL / bloom / FXAA）需要一套可供 demo、集成测试和后续 loader 共同消费的测试资产基线。当前仓库仍将模型和纹理散落在根目录 `models/` 和 `textures/`，没有 `assets/` 顶层目录、没有路径定位 helper、没有构建期同步，文档也把"资产基线已完成"写成了既成事实。如果不先把这层基础设施立住，REQ-011（glTF）、REQ-019（demo 场景）、REQ-028（环境贴图）都无法稳定推进。

## What Changes

- 新建顶层 `assets/` 目录，含 `models/`、`textures/`、`env/` 子目录，迁入现有 `models/viking_room.obj` 和 `textures/viking_room.png`
- 引入 DamagedHelmet（PBR 主测试模型）、Sponza（多 mesh / shadow 压力场景）、studio_small_03_2k.hdr（IBL 环境贴图）、Stanford Bunny（经典 baseline），总量控制在 100 MB 以内
- 每个资产目录附带 `README.md`（来源、license、用途），顶层 `assets/README.md` 汇总清单与体积统计
- 在 `src/core/utils/filesystem_tools` 新增 `cdToWhereAssetsExist(subpath)` helper，从 cwd 向上搜索 `assets/<subpath>`
- 顶层 `CMakeLists.txt` 新增 `assets/` 构建期同步（symlink 优先，copy 兜底）
- 新增 `src/test/integration/test_assets_layout.cpp`，覆盖路径定位 helper 的正/负用例
- 修正 `notes/concepts/assets/index.md` 等文档中"资产基线已完成"的错误描述
- 删除根目录旧 `models/` 和 `textures/`，更新所有引用旧路径的调用点

## Capabilities

### New Capabilities
- `asset-directory-convention`: 测试资产目录结构、体积预算、来源与 license 说明约定
- `asset-path-helper`: `cdToWhereAssetsExist(subpath)` 路径定位 helper 及构建期同步

### Modified Capabilities
- `texture-loading`: 引用路径从 `textures/` 迁移到 `assets/textures/`（路径约定变更）
- `mesh-loading`: 引用路径从 `models/` 迁移到 `assets/models/`（路径约定变更）

## Impact

- **代码**：`filesystem_tools.hpp/.cpp`（新增 helper）、顶层 `CMakeLists.txt`（同步规则）、所有引用 `models/` 或 `textures/` 的源文件
- **测试**：新增 `test_assets_layout`，需在 `src/test/CMakeLists.txt` 注册
- **文档**：`notes/concepts/assets/index.md` 及相关 roadmap / 概念文档
- **仓库结构**：根目录 `models/` 和 `textures/` 被删除，所有资产统一收敛到 `assets/`
- **依赖**：无外部新增依赖
