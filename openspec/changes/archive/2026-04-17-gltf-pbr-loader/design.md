## Context

仓库的 `GLTFLoader` 当前停在桩实现；`load()` 对任何 glTF 文件都抛 `not yet supported`。既有 `ObjLoader` 已经依赖 `tinyobjloader`（单 header，define 宏在消费 .cpp 顶部），`stb_image` 同样是该模式；`SPIRV-Cross` / `yaml-cpp` 是 vendored source tree；`SDL3` 是 vendored 预编译包。仓库**不**使用 submodule / FetchContent / 任何联网下载——`src/infra/CMakeLists.txt` 在 `find_library` / `target_include_directories(external/include)` 之外没有其他第三方获取路径。

`cgltf` 是一个单 header 的 MIT glTF 解析器，非常贴合该模式：头文件放到 `src/infra/external/include/cgltf/cgltf.h`，一个 .cpp 提供 `CGLTF_IMPLEMENTATION`。`REQ-010` 已把 `DamagedHelmet.gltf` + `DamagedHelmet.bin` + 5 张 default_* 贴图落在 `assets/models/damaged_helmet/`；`cdToWhereAssetsExist()` 在 `src/core/utils/filesystem_tools.hpp` 可用。现有 `generic_material_loader` 还没有消费 glTF 材质 metadata 的路径——本 REQ 只暴露，不桥接。

## Goals / Non-Goals

**Goals:**
- `cgltf` 以 vendored header + impl host 方式接入，符合既有第三方依赖模式
- `GLTFLoader::load()` 能真实解析 ASCII `.gltf` + 外部 `.bin` + 外部贴图，消费 `meshes[0].primitives[0]`
- 暴露 POSITION / NORMAL / TEXCOORD_0 / TANGENT (optional) / indices
- `GLTFPbrMaterial` 暴露 base color / metallicRoughness / normal / occlusion / emissive 的 factor + texture path
- 贴图路径以相对 `.gltf` 所在目录暴露
- DamagedHelmet 能通过集成测试 end-to-end 加载

**Non-Goals:**
- 不做节点层级、动画、skin、morph target、scene graph 导入
- 不做多 mesh / 多 primitive 合并
- 不做 tangent 自动生成（无 MikkTSpace fallback）
- 不解析 data URI / base64 嵌入图像
- 不把 `GLTFPbrMaterial` 自动桥接到 `MaterialInstance`
- 不把 `namespace infra` 统一整理到 `LX_infra`
- `.glb` 支持允许实现但不作为首要验收 target

## Decisions

### D1: `cgltf` single-header + 独立 impl host `cgltf_impl.cpp`

**选择**：`src/infra/external/include/cgltf/cgltf.h` + `src/infra/mesh_loader/cgltf_impl.cpp`（内容仅为 `#define CGLTF_IMPLEMENTATION` + `#include "cgltf/cgltf.h"`），后者加入 `INFRA_SOURCES`。

**替代方案**：
- 在 `gltf_mesh_loader.cpp` 顶部直接 `#define CGLTF_IMPLEMENTATION`（与 `obj_mesh_loader.cpp` 风格一致）→ 会在每次触发 `gltf_mesh_loader.cpp` 编译时顺带编 cgltf 实现（几千行），调试迭代变慢；分离 host 把实现编译隔离成一次
- 做成 INTERFACE target → 无源文件可编译，需要把宏定义推给消费方，和项目内其他 vendored lib 的体感不一致

**理由**：REQ 原文直接指定了 impl host 形态；与"一次编译、多次链接"心智匹配；不破坏 `obj_mesh_loader.cpp` 原状。

### D2: PImpl 内聚 cgltf 类型，不让 `cgltf_data*` 逃逸到 header

**选择**：`gltf_mesh_loader.hpp` 继续保持 PImpl 结构，所有 `cgltf_data*` / `cgltf_options` 只出现在 `.cpp` 的 `struct Impl` 与 helper function 里。`Impl` 持有：
```cpp
std::vector<Vec3f> positions;
std::vector<Vec3f> normals;
std::vector<Vec2f> texCoords;
std::vector<Vec4f> tangents;
std::vector<uint32_t> indices;
GLTFPbrMaterial material;
```

**替代方案**：把 cgltf 类型暴露到头文件 → 强制所有消费方引入 cgltf 头，违背已存在的 `namespace infra` PImpl 边界

**理由**：既有 `GLTFLoader` 已经是 PImpl；本 REQ 只是填实内部。消费方（REQ-019 demo）只关心 `std::vector<VecN>` + `GLTFPbrMaterial`。

### D3: 错误传播统一走 `std::runtime_error`，带文件路径 + cgltf 错误码

**选择**：定义内部 helper

```cpp
[[noreturn]] void fail(const std::string& file, const char* what);
[[noreturn]] void failCgltf(const std::string& file, const char* stage,
                            cgltf_result r);
```

`failCgltf` 会把 cgltf 返回码翻译成可读字符串（`cgltf_result_success` → `"success"`, `cgltf_result_invalid_gltf` → `"invalid_gltf"`, 等）。所有失败路径抛 `std::runtime_error("[GLTFLoader <file>] <stage>: <detail>")`。

**替代方案**：返回 `bool` + out param → 与现有 `ObjLoader` / `TextureLoader` 抛异常的风格不一致；加新异常类型 → 调用方要多写 catch

**理由**：一致的错误路径；message 里带 stage 让 debug 容易。

### D4: 贴图路径以"相对 .gltf 所在目录"暴露

**选择**：

```cpp
// 解析时：
std::filesystem::path base = std::filesystem::path(filename).parent_path();
// cgltf_image::uri 是 "Default_albedo.jpg" 这种相对于 gltf 目录的字符串
material.baseColorTexture = image->uri ? std::string(image->uri) : "";
```

`GLTFPbrMaterial::<x>Texture` 里存的就是 URI，调用方自己拼 `base / texture` 或用 `cdToWhereAssetsExist()` 切到 gltf 所在目录后直接以相对路径打开。

**替代方案**：
- 存绝对路径 → 破坏"可迁移 assets/"的假设
- 存 `base + uri` 拼好的完整相对路径 → 调用方仍然要知道 `base`，两次拼接反而更混
- 预先 resolve 文件存在性 → 把 IO 副作用塞进 loader；loader 只负责"告诉你 glTF 怎么声明的"

**理由**：REQ 明确要求"相对 `.gltf` 所在目录"；把 resolve 责任留给消费方和 `cdToWhereAssetsExist()` 已有的机制。

### D5: data URI / base64 嵌入图像直接抛异常，不静默跳过

**选择**：解析 material 时若 `cgltf_image::uri` 为空或以 `data:` 前缀开头（或有 `buffer_view` 指向 buffer 内嵌），抛 `std::runtime_error`，message 带 "data URI images are not supported in this loader"。

**替代方案**：
- 静默用空字符串 → 调用方看到空路径会以为"没有此贴图"，与"有贴图但我们不支持"语义混淆
- 尝试自己 decode base64 → 超出本 REQ 范围

**理由**：REQ R6 明确要求这种情况必须抛；显式早失败比隐式静默更安全。

### D6: 多 mesh / 多 primitive：warning + 取 [0]

**选择**：若 `data->meshes_count > 1` 或 `data->meshes[0].primitives_count > 1`，通过 `std::cerr << "[GLTFLoader] warning: ..."` 打日志并继续用 index 0。不抛异常，不终止加载。

**替代方案**：
- 抛异常 → 直接堵死合法但复杂的资产加载路径（很多 glTF 自带多 mesh）
- 静默 → 不符合 REQ R6 的 "记录 warning" 条款

**理由**：REQ 明确"多 mesh/多 primitive 时：不要求正确合并，可记录 warning 后只取第一个"。

### D7: index 类型兼容

**选择**：索引 accessor 的 component type 允许 `UNSIGNED_BYTE` / `UNSIGNED_SHORT` / `UNSIGNED_INT`，按源类型读出后统一转成 `uint32_t`。其他类型（如 `signed short` 等非法 index 类型）抛异常。

**实现**：

```cpp
switch (accessor->component_type) {
case cgltf_component_type_r_8u:   // widen u8 -> u32
case cgltf_component_type_r_16u:  // widen u16 -> u32
case cgltf_component_type_r_32u:  // copy u32
  // ... use cgltf_accessor_read_index()
default:
  throw ...; // unsupported index type
}
```

**理由**：DamagedHelmet 索引是 u16，REQ-019 demo 场景也常是 u16；兼容性代价是小量拷贝。

### D8: primitive 非三角形直接拒绝

**选择**：`primitive->type != cgltf_primitive_type_triangles` 时抛异常。triangle strip / fan / lines / points 不支持。

**理由**：本工程 pipeline 目前是三角形 topology 打底（`PipelineKey` / `FrameGraph` 都假设三角形）。让 loader 早失败比让 draw 层出奇怪结果好。

### D9: `.glb` 允许但不强制

**选择**：`cgltf_parse_file` 本身就会根据文件扩展名 / magic 自动识别 `.gltf` vs `.glb`。不主动加 `.glb` 特殊分支；如果传进来是 `.glb` 且 cgltf 能解析，照常走。集成测试只用 `.gltf` 路径。

**理由**：REQ 允许 `.glb`；不加特例最小化维护面。

### D10: 测试环境下的资产切换

**选择**：`test_gltf_loader.cpp` 走已有 `cdToWhereAssetsExist("models/damaged_helmet/DamagedHelmet.gltf")` 切入 `assets/` 根，然后 `load("assets/models/damaged_helmet/DamagedHelmet.gltf")`——与 `test_assets_layout` / `test_generic_material_loader` 的调用约定一致。

**替代方案**：直接在 CMake 里传入绝对路径 → 测试与 CWD 强耦合

**理由**：沿用既有模式，不引入第二种资产路径约定。

### D11: DamagedHelmet 断言粒度

**选择**：

```cpp
// positions / normals / texCoords / indices 非空，且 indices.size() % 3 == 0
// tangents：若 DamagedHelmet.gltf 声明了 TANGENT（实际上它有），则非空；否则允许空
// material 断言：
//   baseColorTexture 非空且后缀 .jpg / .jpeg / .png 之一
//   metallicRoughnessTexture 非空
//   normalTexture 非空
//   occlusionTexture 非空
//   emissiveTexture 非空
//   baseColorFactor.w > 0.0f （sanity：通常 1.0f）
```

不断言具体文件名字符串（REQ 明确说"不应把具体贴图文件名写死到过于脆弱的程度"）。

**理由**：DamagedHelmet 是标准 PBR 测试资产，5 张贴图都有；把"存在"作为 invariant 比把"叫什么"作为 invariant 更稳。

## Risks / Trade-offs

- **[cgltf header 版本与仓库生命周期]** → vendored header 可能过时；缓解：`external/README.md` 记录引入的 commit / version，让后续升级有迹可循；cgltf MIT 许可，升级只需替换一个文件
- **[impl host cpp 把 cgltf 内部 warning 变成项目 warning]** → cgltf 有已知的 signed-unsigned / fallthrough warning；缓解：如有需要在 `cgltf_impl.cpp` 前置 `#pragma GCC diagnostic push/ignored` 控制作用域只限该文件
- **[DamagedHelmet 资产授权 / 版本不一致]** → REQ-010 落地的资产应已完成 license 核对；本 REQ 不重新介入
- **[`GLTFPbrMaterial` 使用 `std::string` 字段]** → 5 个 texture 字段 × 短字符串，没有性能或内存问题；但 PImpl 的 `getMaterial()` 返回 const ref 到 Impl 内部，调用方不应持有此 ref 越过下一次 `load()`——这是 PImpl 通用约束，不需要特殊文档
- **[缺 TANGENT 但有 normal map 的 glTF]** → 本 REQ 返回空 tangent vector；消费端（REQ-019 / 材质桥接）需要自己判断是否需要 reject 或 fallback；风险显式留给上层
- **[测试二进制没有 assets 拷贝]** → `test_assets_layout` 已验证 asset 路径收敛；`test_gltf_loader` 沿用同机制

## Migration Plan

1. vendored `cgltf.h` + 新建/扩充 `external/README.md`
2. 新建 `cgltf_impl.cpp` 并加入 `INFRA_SOURCES`；先 build 通过验证依赖接入正确
3. 扩展 `gltf_mesh_loader.hpp`：增加 `GLTFPbrMaterial`、`getTangents`、`getMaterial`
4. 重写 `gltf_mesh_loader.cpp`：cgltf 解析 + attribute 抽取 + material 抽取 + 错误处理
5. 新增 `test_gltf_loader.cpp` + CMake 注册，跑 DamagedHelmet 端到端
6. 同步 `notes/concepts/assets/index.md`（若存在"glTF 已承载 PBR 元数据"先行描述）

每一步可独立 build；失败不会堵到未接入 loader 的既有测试。

## Open Questions

- 是否把 cgltf 编译 warning 抑制写进 `cgltf_impl.cpp` 还是让它暴露 → 实现阶段视首次 build 真实 warning 决定；若仓库 `-Wall -Werror` 触发就抑制，否则不加
- `GLTFPbrMaterial` 的 `emissiveFactor` 用 `Vec3f` 与 `emissiveTexture` 缺失时如何判定 → 默认 `{0,0,0}` + 空 texture 字符串即为"没有 emissive"；与 glTF spec 天然一致，不需要额外标志位
