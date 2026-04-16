## 1. 资产目录与文件迁移

- [x] 1.1 创建 `assets/models/`、`assets/textures/`、`assets/env/` 目录结构
- [x] 1.2 将 `models/viking_room.obj` 迁移到 `assets/models/viking_room/viking_room.obj`
- [x] 1.3 将 `textures/viking_room.png` 迁移到 `assets/textures/viking_room/viking_room.png`
- [x] 1.4 下载并放置 DamagedHelmet glTF 资产到 `assets/models/damaged_helmet/`
- [x] 1.5 下载并放置 Sponza glTF 资产到 `assets/models/sponza/`
- [x] 1.6 下载并放置 studio_small_03_2k.hdr 到 `assets/env/`
- [x] 1.7 下载并放置 Stanford Bunny 到 `assets/models/stanford_bunny/`（若超 100 MB 预算则按裁剪策略处理）
- [x] 1.8 删除根目录 `models/` 和 `textures/` 目录
- [x] 1.9 验证 `assets/` 总大小 <= 100 MB

## 2. 资产说明文档

- [x] 2.1 为 `assets/models/damaged_helmet/` 编写 README.md（来源、license、文件列表、体积）
- [x] 2.2 为 `assets/models/sponza/` 编写 README.md
- [x] 2.3 为 `assets/models/stanford_bunny/` 编写 README.md
- [x] 2.4 为 `assets/models/viking_room/` 编写 README.md
- [x] 2.5 为 `assets/textures/viking_room/` 编写 README.md
- [x] 2.6 为 `assets/env/` 编写 README.md
- [x] 2.7 编写 `assets/README.md` 总览（全部资产清单、总体积统计、裁剪原则、下游需求）

## 3. 路径定位 Helper

- [x] 3.1 在 `src/core/utils/filesystem_tools.hpp` 新增 `cdToWhereAssetsExist(const std::string& subpath)` 声明
- [x] 3.2 在 `src/core/utils/filesystem_tools.cpp` 实现 `cdToWhereAssetsExist()`（向上搜索 8 级，匹配 `assets/<subpath>`）
- [x] 3.3 更新所有引用旧 `models/` 或 `textures/` 路径的代码（如有）

## 4. CMake 构建期同步

- [x] 4.1 在顶层 `CMakeLists.txt` 添加 `assets/` symlink 创建规则（`create_symlink`）
- [x] 4.2 添加 symlink 失败时的 copy 兜底逻辑
- [x] 4.3 验证从 build 目录可访问 `assets/` 下的资产

## 5. 集成测试

- [x] 5.1 创建 `src/test/integration/test_assets_layout.cpp`，覆盖 4 个测试用例（DamagedHelmet、HDR、viking_room 正向 + nonexistent 负向）
- [x] 5.2 在 `src/test/CMakeLists.txt` 注册 `test_assets_layout` 测试
- [x] 5.3 从源码目录和 build 目录分别运行测试，验证均通过

## 6. 文档修正

- [x] 6.1 修正 `notes/concepts/assets/index.md` 中"已实现"描述为实际状态
- [x] 6.2 检查并修正其他引用旧 `models/` / `textures/` 路径的文档
- [x] 6.3 确认文档与实际仓库目录、helper、测试覆盖一致
