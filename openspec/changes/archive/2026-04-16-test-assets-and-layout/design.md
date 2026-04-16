## Context

仓库当前状态：

- 根目录 `models/viking_room.obj`、`textures/viking_room.png` 和 `textures/texture.jpg` 三个文件散落在顶层
- 无 `assets/` 目录，无路径定位 helper，无构建期同步
- `notes/concepts/assets/index.md` 第 83 行把 `assets/` 目录约定和测试资产基线写成"已实现"，与实际不符
- `filesystem_tools.cpp` 只有 `cdToWhereShadersExist()`，从 cwd 向上最多搜索 8 级
- 代码中暂未发现直接硬编码 `models/` 或 `textures/` 的引用（demo 入口可能通过相对路径消费，需在实现时确认）

## Goals / Non-Goals

**Goals:**
- 建立 `assets/` 顶层目录，将所有测试资产收敛到统一位置
- 引入 DamagedHelmet、Sponza、HDR 环境贴图、Stanford Bunny，满足 Phase 1 下游需求
- 提供 `cdToWhereAssetsExist(subpath)` helper，与现有 `cdToWhereShadersExist()` 风格一致
- CMake 构建期同步，使 build 目录也能访问 `assets/`
- 修正文档中"资产基线已完成"的错误描述

**Non-Goals:**
- 不实现任何 loader（OBJ / glTF / HDR 的解析属于各自的 REQ）
- 不引入统一 `AssetManager` 或资产注册表
- 不引入 git LFS 或 git submodule
- 不修改渲染逻辑
- 不重构 `filesystem_tools` 的命名空间或接口风格

## Decisions

### D1: 资产直接提交到仓库，不使用 LFS

**选择**：将测试资产（约 65 MB + bunny）直接提交到 git 仓库。

**替代方案**：
- git LFS：增加 CI 复杂度，需要所有开发者配置 LFS
- 下载脚本：不可离线工作，增加 setup 步骤

**理由**：总量 < 100 MB，对 git 可接受；无外部依赖，clone 即可用。

### D2: `cdToWhereAssetsExist()` 沿用 `cdToWhereShadersExist()` 的搜索模式

**选择**：从 cwd 向上搜索最多 8 级，匹配 `<dir>/assets/<subpath>` 存在性，找到后切换 cwd。

**替代方案**：
- 返回路径而不切换 cwd：与现有 shader helper 不一致
- 环境变量指定 assets root：增加配置负担

**理由**：与现有代码风格一致，最小改动。命名空间收敛留给独立整理需求。

### D3: CMake 同步使用 symlink 优先，copy 兜底

**选择**：在顶层 `CMakeLists.txt` 中使用 `create_symlink` 将 `${CMAKE_SOURCE_DIR}/assets` 链接到 `${CMAKE_BINARY_DIR}/assets`，Windows 或 symlink 不可用时回退到 `file(COPY ...)`。

**替代方案**：
- 每次构建全量 copy：浪费时间和磁盘
- 不做同步，完全依赖 helper 向上搜索：build 目录下直接访问资产更方便

**理由**：symlink 零开销，大部分平台支持；兜底保证 Windows 兼容。

### D4: 每个资产目录独立 README.md

**选择**：每个顶级资产目录（如 `assets/models/damaged_helmet/`）放一个 `README.md`，记录来源、license、文件列表、体积等。`assets/README.md` 作为总览。

**理由**：资产来源可追溯，license 合规，不依赖外部数据库。

### D5: Stanford Bunny 裁剪策略

**选择**：优先选用轻量 `.obj` 版本。若加入后总量超 100 MB，按 REQ-010 裁剪顺序处理（替换更小版本 → 移除并记录原因）。DamagedHelmet / Sponza / HDR / viking_room 不可移除。

## Risks / Trade-offs

- **[仓库体积增长]** → 100 MB 硬上限，且 REQ 明确禁止 4K HDR 和额外大场景。后续如需更多资产，可引入 LFS 或下载脚本。
- **[Sponza 文件数量多]** → 可能增加 clone 时间，但作为多 mesh 压力测试场景不可替代。
- **[symlink 在 Windows 上需要权限]** → copy 兜底方案已覆盖。
- **[`cdToWhereAssetsExist()` 改变 cwd 是全局副作用]** → 与现有 shader helper 行为一致，当前项目规模下可接受。后续如需并发安全，统一重构。
