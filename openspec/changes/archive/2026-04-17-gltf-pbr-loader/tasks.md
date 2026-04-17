## 1. Vendored cgltf 接入

- [x] 1.1 下载/落地 `cgltf` single-header 到 `src/infra/external/include/cgltf/cgltf.h`（MIT 许可，来源 https://github.com/jkuhlmann/cgltf，v1.15，7175 行）；构建流程本身无联网步骤（头文件随仓库提交）
- [x] 1.2 新建 `src/infra/external/README.md`：列出全部 vendored 依赖 (cgltf, stb, tinyobjloader, imgui, SDL3, SPIRV-Cross, yaml-cpp) 的 shape / upstream / version / license / consumers
- [x] 1.3 新建 `src/infra/mesh_loader/cgltf_impl.cpp`，内容仅为 `#define CGLTF_IMPLEMENTATION` + `#include "cgltf/cgltf.h"`
- [x] 1.4 把 `mesh_loader/cgltf_impl.cpp` 追加到 `src/infra/CMakeLists.txt` 的 `INFRA_SOURCES`；首次 build 无 warning，暂不加 `#pragma` 抑制
- [x] 1.5 `cmake --build build --target LX_Infra` 通过，cgltf_impl.cpp 编译链接无误

## 2. Header 扩展

- [x] 2.1 在 `src/infra/mesh_loader/gltf_mesh_loader.hpp` 增加 `struct GLTFPbrMaterial`，默认值 `{1,1,1,1}` base / `1.0f` metallic / `1.0f` roughness / `{0,0,0}` emissive / 空 texture 字符串
- [x] 2.2 给 `GLTFLoader` 追加 `const std::vector<LX_core::Vec4f>& getTangents() const;` 与 `const GLTFPbrMaterial& getMaterial() const;`
- [x] 2.3 保持 PImpl 结构；hpp 只 include `core/math/vec.hpp` + `<string>` + `<vector>`，不引入 cgltf

## 3. Loader 真实实现

- [x] 3.1 更新 `gltf_mesh_loader.cpp` 的 `struct Impl`：添加 `std::vector<Vec4f> tangents;` 与 `GLTFPbrMaterial material;`
- [x] 3.2 重写 `GLTFLoader::load`：`cgltf_options{}` + `cgltf_parse_file` + `cgltf_load_buffers`，失败抛 runtime_error 并带文件路径与 cgltf_result 可读化；`std::unique_ptr<cgltf_data, CgltfDeleter>` 保证 `cgltf_free` 在所有返回路径都执行
- [x] 3.3 选 primitive：`meshes_count > 1` / `primitives_count > 1` 打 cerr warning + 用 [0]；`meshes_count == 0` / `primitives_count == 0` 抛异常
- [x] 3.4 抽取 attributes：POSITION 必需（缺失抛）；NORMAL / TEXCOORD_0 / TANGENT 可选；统一走 `cgltf_accessor_read_float` 填 Vec3f/Vec2f/Vec4f vector
- [x] 3.5 抽取 indices：primitive 非三角形抛；component_type 非 u8/u16/u32 抛；`cgltf_accessor_read_index` widen 到 uint32_t；无 indices 时按顺序合成
- [x] 3.6 抽取 material：空 material 保留默认；否则从 `pbr_metallic_roughness` 读 factor + baseColor/metallicRoughness texture；从 material 根读 emissive_factor / normal / occlusion / emissive texture
- [x] 3.7 贴图 URI：`image->uri == nullptr && buffer_view != nullptr` 抛（inline/buffer_view 不支持）；URI 以 `data:` 开头抛；否则原样存字符串（相对 gltf 目录）
- [x] 3.8 实现 `getTangents()` / `getMaterial()`

## 4. 集成测试

- [x] 4.1 新建 `src/test/integration/test_gltf_loader.cpp`，遵循既有 `[FAIL]` / `[PASS]` 测试风格；include `cdToWhereAssetsExist` + `infra/mesh_loader/gltf_mesh_loader.hpp`
- [x] 4.2 用例 `loads_damaged_helmet`：断言几何流非空且 `indices.size() % 3 == 0`；断言 `getTangents()` **为空**（本仓库 DamagedHelmet.gltf 未声明 TANGENT，loader 不得抛异常）；断言 5 张 PBR texture URI 均非空（不断言具体文件名）
- [x] 4.3 用例 `throws_on_missing_file`：对不存在的路径调用 `load` 并断言 `std::runtime_error` + message 含文件名
- [x] 4.4 用例 `throws_on_corrupt_file`：`load("assets/models/viking_room/viking_room.obj")` 断言抛出
- [x] 4.5 在 `src/test/CMakeLists.txt` 把 `test_gltf_loader` 加入 `TEST_INTEGRATION_EXE_LIST`
- [x] 4.6 本地构建 + 运行：`cmake --build build --target test_gltf_loader && ./build/src/test/test_gltf_loader` PASS

## 5. 收尾

- [x] 5.1 `cmake --build build` 全量无回归；`test_input_state` / `test_sdl_input` / `test_imgui_overlay` / `test_debug_ui_smoke` / `test_gltf_loader` / `test_assets_layout` / `test_engine_loop` 全部 PASS
- [x] 5.2 核查 `notes/concepts/assets/index.md`：现有描述"GLTF 路径已经能承载 PBR 相关元数据"先前是 aspirational，现随 REQ-011 落地而成立，文字无需改动
- [x] 5.3 更新 `docs/requirements/011-gltf-pbr-loader.md` 的 `## 实施状态` 段落
