# REQ-010: 测试用 3D 资产清单与 assets/ 目录约定

## 背景

为了支持 [Phase 1 渲染深度](../../notes/roadmaps/phase-1-rendering-depth.md)（PBR / shadow / IBL / bloom / FXAA），调试链路需要一组**图形学社区公认**的测试资产 —— 当前仓库只有 `models/viking_room.obj` + `textures/viking_room.png` 两个文件，且 `models/` / `textures/` 直接放在仓库根目录，缺少：

- PBR 全通道（baseColor / metalRoughness / normal / AO / emissive）的参考模型
- 多 mesh + 多材质的"压力测试"场景（用于 frustum culling、draw call 性能、shadow 覆盖）
- HDR equirectangular 环境贴图（IBL 预过滤 REQ-105 / REQ-106 的输入）
- 资产来源 / license 标注 —— 任何引入仓库的资源都应当可追溯
- 一个统一的 `cdToWhereAssetsExist()` 入口，跟现有 `cdToWhereShadersExist()`（`src/test/test_render_triangle.cpp:35`）配对

本需求**只搬运资产 + 建目录约定 + 写工具函数**，不实现任何 loader。glTF 文件的实际解析见 REQ-011。

## 目标

1. 仓库新增 `assets/` 顶层目录，按子领域分层；`models/` / `textures/` 两个旧根目录的内容迁入
2. 引入一组图形学界通用的 PBR / shadow / IBL 测试资产并附 license
3. 提供 `cdToWhereAssetsExist(subpath)` 工具函数，与现有 `cdToWhereShadersExist` 风格一致
4. CMake 在构建期把 `assets/` 同步到 build 目录，让 demo / 集成测试就地访问

## 需求

### R1: `assets/` 目录约定

新建顶层 `assets/`，结构如下：

```
assets/
├── models/
│   ├── damaged_helmet/         # Khronos glTF Sample Models, CC-BY 4.0
│   │   ├── DamagedHelmet.gltf
│   │   ├── DamagedHelmet.bin
│   │   ├── *.png (5 张 PBR 贴图)
│   │   └── README.md
│   ├── sponza/                 # Crytek/Intel Sponza, CC0/CC-BY
│   │   ├── Sponza.gltf
│   │   ├── ...
│   │   └── README.md
│   ├── stanford_bunny/         # Stanford 3D Scanning Repository
│   │   ├── bunny.obj
│   │   └── README.md
│   └── viking_room/            # 当前 models/viking_room.obj 迁入
│       ├── viking_room.obj
│       └── README.md
├── textures/
│   ├── viking_room/
│   │   └── viking_room.png    # 当前 textures/viking_room.png 迁入
│   └── (未来 PBR 单独贴图组)
└── env/
    ├── studio_small_03_2k.hdr  # PolyHaven CC0
    └── README.md
```

每个子目录的 `README.md` 必须写明：

- 资产名 + 来源 URL
- 原始 license（CC0 / CC-BY 4.0 / Stanford 学术使用条款 等）
- 本仓库使用方式说明（"REQ-105 的 IBL 预过滤输入"）
- 文件大小 + 三角面数（让后续 REQ 估算性能）

仓库根目录的旧 `models/` / `textures/` 删除，所有引用更新到新路径。

### R2: 资产清单（最小集）

本 REQ 仅引入下列资产，体积控制在 **总 ≤ 80 MB**：

| 资产 | 用途 | 体积估算 |
|---|---|---|
| **DamagedHelmet** (Khronos glTF Sample Models) | PBR 调试主力、IBL 验证 | ~3 MB |
| **Sponza** (Crytek/Intel glTF 版) | shadow / 多 mesh / culling 压力 | ~50 MB |
| **Stanford Bunny** (`.obj`) | 经典 baseline、shader 反射验证 | ~1 MB |
| **studio_small_03_2k.hdr** (PolyHaven) | IBL 环境光输入 | ~5 MB |
| viking_room (现有) | 兼容老 demo | ~0.5 MB |

不在本 REQ 中的：

- Cornell Box（如未来需要 GI 验证再加，Phase 1 不做 GI）
- 高分辨率（4K+）HDR 环境图（Phase 1 暂用 2K）
- Lumberyard Bistro / Amazon Bistro（Phase 3 资产管线再考虑）

### R3: `cdToWhereAssetsExist(subpath)` helper

在 `src/core/utils/filesystem_tools.hpp` 增加（与现有 `cdToWhereShadersExist` 同文件）：

```cpp
namespace LX_core {

/// 沿当前工作目录向上搜索，定位到包含 `assets/<subpath>` 的目录后切换 cwd 到该目录。
/// 用法：
///   if (!cdToWhereAssetsExist("models/damaged_helmet/DamagedHelmet.gltf")) {
///     std::cerr << "asset not found"; return 1;
///   }
/// 成功后调用方可直接以 "assets/models/damaged_helmet/DamagedHelmet.gltf" 相对路径打开。
bool cdToWhereAssetsExist(const std::string &subpath);

}
```

实现策略与 `cdToWhereShadersExist` 一致：从 cwd 起向上最多 N 层，逐层检查 `<dir>/assets/<subpath>` 是否存在。

### R4: CMake 资产同步

在顶层 `CMakeLists.txt` 新增一条 custom command / install rule，把 `assets/` 目录在构建期通过 `file(COPY ...)` 或 symlink 的方式同步到 `${CMAKE_BINARY_DIR}/assets/`。

约束：

- 不要在每次 build 都全量 copy（用 `add_custom_target` 配合 mtime 判断，或者建 symlink）
- 单元测试 / 集成测试如果跑在 `${CMAKE_BINARY_DIR}` 之外（如 ctest 临时 dir）也要能找到 —— `cdToWhereAssetsExist` 兜底向上搜
- Windows 不要硬依赖 symlink

## 测试

- 新建 `src/test/integration/test_assets_layout.cpp`：
  - 断言 `cdToWhereAssetsExist("models/damaged_helmet/DamagedHelmet.gltf")` 返回 `true`
  - 断言 `cdToWhereAssetsExist("env/studio_small_03_2k.hdr")` 返回 `true`
  - 断言 `cdToWhereAssetsExist("nonexistent/foo.bar")` 返回 `false`
- 不验证文件内容 —— 那是 REQ-011 / REQ-105 的事

## 修改范围

| 文件 | 改动 |
|---|---|
| `assets/**` | 新增（含 README + license） |
| `models/**` | 删除（迁入 `assets/models/`） |
| `textures/**` | 删除（迁入 `assets/textures/` 或对应 `assets/models/<name>/`） |
| `src/core/utils/filesystem_tools.hpp` | 新增 `cdToWhereAssetsExist` 声明 |
| `src/core/utils/filesystem_tools.cpp` | 新增实现 |
| `src/test/test_render_triangle.cpp:35` | 旧的 `viking_room` 路径引用更新为 `assets/models/viking_room/...` |
| `CMakeLists.txt` | 资产同步规则 |
| `src/test/integration/test_assets_layout.cpp` | 新增 |
| `src/test/integration/CMakeLists.txt` | 注册新测试 |

## 边界与约束

- **不实现任何 loader** —— glTF 的实际解析见 REQ-011；HDR 环境贴图的解析见 REQ-105
- **不下载到 build dir** —— 资产作为仓库一部分，git LFS 与否由后续决定。本 REQ 默认走普通 git，需要用户接受 ~80 MB 的初始 clone 体积
- 不引入 git submodule —— Khronos glTF Sample Models 库整体太大，只挑 DamagedHelmet 单模型 copy 进来
- 不修改任何渲染代码

## 依赖

- 无（基础设施类需求）

## 下游

- **REQ-011**：glTF PBR loader 用本 REQ 提供的 DamagedHelmet 作为唯一非平凡测试输入
- **REQ-019**：demo_scene_viewer 加载 DamagedHelmet + Sponza
- **REQ-105 / REQ-106**：环境贴图加载与 IBL 预过滤直接读 `assets/env/*.hdr`
- **REQ-110**：frustum culling 用 Sponza 作为压力测试场景

## 实施状态

2026-04-16 核查结果：未开始。

- 仓库仍使用根目录 `models/` 与 `textures/`
- 尚无 `assets/` 顶层目录
- 尚无 `cdToWhereAssetsExist()` helper
