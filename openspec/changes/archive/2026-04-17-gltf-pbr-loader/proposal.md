## Why

`GLTFLoader::load()` 当前是桩实现：读 12 字节 magic 就对 `.gltf` / `.glb` 一律抛 `not yet supported`。Phase 1 PBR 调试链路（REQ-019 demo scene viewer、后续材质桥接）需要真实加载 `DamagedHelmet` 并拿到 几何流 + tangent + PBR 元数据；`REQ-010` 已经把资产备好，但没有 loader 消费它。仓库当前没有任何 vendored glTF 解析库，而第三方依赖接入模式已经固化为"直接入仓 + 离线构建"（`stb` / `tinyobjloader` / `SDL3` / `SPIRV-Cross` / `yaml-cpp` 均已按此模式接入）。本变更把 `cgltf` 以该模式引入，并把 loader 提升到能产出 `DamagedHelmet` 几何 + PBR 元数据的程度。

## What Changes

- 在 `src/infra/external/include/cgltf/cgltf.h` vendored `cgltf` single-header（不使用 submodule / FetchContent / 联网下载）
- 新增 `src/infra/external/README.md`（若不存在）登记 cgltf 来源、版本与 MIT license；若已存在则补 cgltf 条目
- 新增 `src/infra/mesh_loader/cgltf_impl.cpp`，仅含 `#define CGLTF_IMPLEMENTATION` + `#include "cgltf/cgltf.h"`；加入 `src/infra/CMakeLists.txt` 的 `INFRA_SOURCES`
- **BREAKING**（仅接口扩展，调用方只增不减）：扩展 `src/infra/mesh_loader/gltf_mesh_loader.hpp`：
  - 新增 `struct GLTFPbrMaterial { Vec4f baseColorFactor; float metallicFactor; float roughnessFactor; Vec3f emissiveFactor; std::string baseColorTexture / metallicRoughnessTexture / normalTexture / occlusionTexture / emissiveTexture; }`
  - 新增 `const std::vector<Vec4f>& getTangents() const`
  - 新增 `const GLTFPbrMaterial& getMaterial() const`
- 重写 `gltf_mesh_loader.cpp`：用 `cgltf_parse_file` + `cgltf_load_buffers` 解析；默认消费 `data->meshes[0].primitives[0]`；提取 POSITION / NORMAL / TEXCOORD_0 / TANGENT（可选）/ indices；从 `primitive->material` 填 `GLTFPbrMaterial`；贴图路径以相对 `.gltf` 所在目录的形式暴露
- 错误处理：文件不存在、`cgltf_parse_file/load_buffers` 失败、缺 POSITION、primitive 非三角形、index 类型不支持、使用 data URI/base64 内嵌图像 均抛 `std::runtime_error`，message 带文件路径与 cgltf 错误码
- 多 mesh / 多 primitive / `.glb` / `.gltf` 带 tangent 的行为：多 mesh/primitive 只取第一个并打 warning（非硬错误）；`.glb` 允许实现但不作为首要验收；缺 TANGENT 时 `getTangents()` 返回空 vector，不做 MikkTSpace 生成
- 新增 `src/test/integration/test_gltf_loader.cpp`：`loads_damaged_helmet` / `throws_on_missing_file` / `throws_on_corrupt_file`，注册到 `src/test/CMakeLists.txt`
- 同步 `notes/concepts/assets/index.md` 等提到"glTF 已承载 PBR 元数据"的文档描述（若落后于现状）

## Capabilities

### New Capabilities

- `gltf-pbr-loader`: `cgltf`-backed 真实 glTF 解析、PBR 材质元数据 (`GLTFPbrMaterial`)、tangent 暴露、DamagedHelmet 集成测试闭环；以及第三方依赖"直接入仓 + 离线构建"约束在 glTF loader 场景下的明确定义

### Modified Capabilities

- `mesh-loading`: `GLTFLoader::load` 从桩实现改为 `cgltf` 真实解析；`GLTFLoader` 的数据访问接口扩展加入 `getTangents()` 与 `getMaterial()`

## Impact

- **代码**：`src/infra/external/include/cgltf/cgltf.h` 新增；`src/infra/external/README.md` 新增或增补 cgltf 条目；`src/infra/mesh_loader/{gltf_mesh_loader.hpp,gltf_mesh_loader.cpp,cgltf_impl.cpp}` 改写/新增；`src/infra/CMakeLists.txt` 加一行
- **构建**：无新依赖（cgltf 纯 header + impl host cpp），无新 find_package / FetchContent / submodule；`external/include` 已在 PRIVATE include path 上
- **测试**：新增 `test_gltf_loader`（3 个用例），注册到 `src/test/CMakeLists.txt`
- **依赖**：`REQ-010`（DamagedHelmet 资产 + `cdToWhereAssetsExist()`，已落地）
- **下游**：`REQ-019` demo scene viewer；后续材质桥接把 `GLTFPbrMaterial` 转成 `MaterialInstance`；Phase 3 资产管线
- **非目标**：节点层级 / 动画 / skin / morph target / 多 mesh 合并 / 完整 scene graph / tangent 自动生成 / data URI 图像解析 / 材质系统自动桥接 / 把 `namespace infra` 统一到 `LX_infra`（命名空间整理不作为本 REQ 闭环阻塞项）
