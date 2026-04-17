## ADDED Requirements

### Requirement: cgltf is vendored under src/infra/external

The `cgltf` single-header glTF parser SHALL be vendored at `src/infra/external/include/cgltf/cgltf.h`. The repository SHALL NOT fetch `cgltf` via git submodule, CMake `FetchContent`, package managers, or any online download at configure or build time. A separate implementation host file at `src/infra/mesh_loader/cgltf_impl.cpp` SHALL contain only:

```cpp
#define CGLTF_IMPLEMENTATION
#include "cgltf/cgltf.h"
```

and SHALL be compiled as part of the `LX_Infra` library via `INFRA_SOURCES`. `src/infra/external/README.md` SHALL list `cgltf` alongside other vendored dependencies, recording the upstream source URL, version / commit, and MIT license.

#### Scenario: Offline build succeeds

- **WHEN** running `cmake` and `cmake --build` in an environment without network access
- **THEN** configuration and build SHALL succeed for `LX_Infra` (including the gltf loader translation units) without downloading anything

#### Scenario: cgltf header is reachable via project include path

- **WHEN** a translation unit under `src/infra/mesh_loader/` writes `#include "cgltf/cgltf.h"`
- **THEN** the include SHALL resolve to `src/infra/external/include/cgltf/cgltf.h` via the existing `external/include` private include directory

#### Scenario: Only one translation unit defines CGLTF_IMPLEMENTATION

- **WHEN** grepping the project for `CGLTF_IMPLEMENTATION`
- **THEN** exactly one `#define CGLTF_IMPLEMENTATION` SHALL appear in the repository, inside `src/infra/mesh_loader/cgltf_impl.cpp`

### Requirement: Vendored-dependency policy for glTF loader

The glTF loader SHALL abide by the repository's established third-party dependency policy:

- **Allowed:** vendored header, vendored source tree, vendored pre-built package with headers + platform libraries
- **Disallowed:** git submodule, CMake `FetchContent`, network download during first configure, system package manager as the only source of truth

The `cgltf` dependency SHALL satisfy the "vendored header + one implementation host cpp" shape.

#### Scenario: No submodule / no FetchContent

- **WHEN** inspecting `.gitmodules` and every `CMakeLists.txt`
- **THEN** no submodule SHALL exist for `cgltf` and no `FetchContent_Declare(cgltf ...)` call SHALL appear

### Requirement: GLTFLoader exposes PBR material metadata struct

The header `src/infra/mesh_loader/gltf_mesh_loader.hpp` SHALL declare:

```cpp
struct GLTFPbrMaterial {
  LX_core::Vec4f baseColorFactor{1.0f, 1.0f, 1.0f, 1.0f};
  float metallicFactor = 1.0f;
  float roughnessFactor = 1.0f;
  LX_core::Vec3f emissiveFactor{0.0f, 0.0f, 0.0f};

  std::string baseColorTexture;
  std::string metallicRoughnessTexture;
  std::string normalTexture;
  std::string occlusionTexture;
  std::string emissiveTexture;
};
```

The struct SHALL live in `namespace infra` (consistent with current `GLTFLoader`). Texture string fields SHALL hold paths relative to the directory containing the `.gltf` file, as declared by the glTF `uri` field, without any filesystem resolution.

#### Scenario: Missing texture field is empty

- **WHEN** a glTF material does not declare a particular texture (e.g. no emissive texture)
- **THEN** the corresponding `GLTFPbrMaterial::<x>Texture` field SHALL be an empty string after `load()` returns

#### Scenario: Default factors when material is absent

- **WHEN** a glTF primitive has no `material` binding
- **THEN** `GLTFPbrMaterial` SHALL take its declared default values (white base color, metallic=1, roughness=1, zero emissive, empty texture strings)

### Requirement: GLTFLoader real parsing via cgltf

`GLTFLoader::load(filename)` SHALL invoke `cgltf_parse_file` followed by `cgltf_load_buffers` to parse the glTF document and fetch external `.bin` buffers. ASCII `.gltf` with external `.bin` and external image files SHALL be supported as the primary path. `.glb` MAY work if cgltf parses it, but SHALL NOT be required for acceptance. The loader SHALL consume `data->meshes[0].primitives[0]` as the mesh source.

Attributes extracted from that primitive SHALL include:

- `POSITION` → `std::vector<Vec3f>` accessible via `getPositions()`
- `NORMAL` → `std::vector<Vec3f>` accessible via `getNormals()`
- `TEXCOORD_0` → `std::vector<Vec2f>` accessible via `getTexCoords()`
- `TANGENT` → `std::vector<Vec4f>` accessible via `getTangents()` (empty when missing)
- indices → `std::vector<uint32_t>` accessible via `getIndices()`

Index accessor component types `UNSIGNED_BYTE`, `UNSIGNED_SHORT`, and `UNSIGNED_INT` SHALL all be widened to `uint32_t` on read.

#### Scenario: Successful parse of ASCII glTF with external bin

- **WHEN** `load("DamagedHelmet.gltf")` is called with a valid ASCII glTF + `.bin`
- **THEN** positions / normals / texCoords / indices SHALL all be non-empty and `indices.size() % 3 == 0`

#### Scenario: Missing TANGENT stays empty

- **WHEN** `load()` succeeds on a glTF without a `TANGENT` accessor
- **THEN** `getTangents()` SHALL return an empty vector and `load()` SHALL NOT throw for this reason

### Requirement: Texture URI paths are relative to the .gltf directory

Each non-empty `GLTFPbrMaterial::<x>Texture` SHALL be the URI declared by the glTF image entry (e.g. `"Default_albedo.jpg"`), relative to the directory that holds the `.gltf` file. The loader SHALL NOT resolve these URIs to absolute paths and SHALL NOT verify that the files exist on disk.

#### Scenario: Relative URI is preserved as-is

- **WHEN** a glTF material binds `baseColorTexture` to an image with `uri = "Default_albedo.jpg"`
- **THEN** `GLTFPbrMaterial::baseColorTexture` SHALL equal `"Default_albedo.jpg"` after `load()`

### Requirement: Multi-mesh and multi-primitive fallback

When `data->meshes_count > 1` or `meshes[0].primitives_count > 1`, the loader SHALL emit a warning (for example to `std::cerr`) that names the file and reports the surplus count, and SHALL continue loading with `meshes[0].primitives[0]`. This condition SHALL NOT be treated as a hard error and SHALL NOT throw.

#### Scenario: Multi-primitive glTF loads the first primitive

- **WHEN** `load()` is called on a glTF whose `meshes[0]` has 3 primitives
- **THEN** `load()` SHALL return without throwing and SHALL populate geometry from `primitives[0]` only

### Requirement: Error handling throws std::runtime_error with context

The loader SHALL throw `std::runtime_error` for each of the following conditions, with a message that contains the input filename and (where applicable) a readable cgltf error code:

1. File does not exist
2. `cgltf_parse_file` fails
3. `cgltf_load_buffers` fails
4. The selected primitive lacks `POSITION`
5. `primitive->type` is not `cgltf_primitive_type_triangles`
6. Index component type is not one of `UNSIGNED_BYTE` / `UNSIGNED_SHORT` / `UNSIGNED_INT`
7. A required image is declared via a `data:` URI or via `buffer_view` (inline / base64-encoded image), which this loader does not support

#### Scenario: Missing file throws with path in message

- **WHEN** `load("does_not_exist.gltf")` is called and the file cannot be opened
- **THEN** `std::runtime_error` SHALL be thrown and its `what()` SHALL contain `"does_not_exist.gltf"`

#### Scenario: Corrupt file throws

- **WHEN** `load()` is called on a file whose contents are not valid glTF (e.g. an `.obj` file re-labelled)
- **THEN** `std::runtime_error` SHALL be thrown from the parse stage

#### Scenario: Non-triangle primitive throws

- **WHEN** `load()` is called on a glTF whose `primitives[0].type` is not triangles
- **THEN** `std::runtime_error` SHALL be thrown with a message naming the file and referring to the primitive type

### Requirement: Integration test covers DamagedHelmet end-to-end

A new test `src/test/integration/test_gltf_loader.cpp` SHALL be added and registered in `src/test/CMakeLists.txt`. It SHALL include:

- `loads_damaged_helmet`
  - `cdToWhereAssetsExist("models/damaged_helmet/DamagedHelmet.gltf")` followed by `load("assets/models/damaged_helmet/DamagedHelmet.gltf")`
  - Asserts `positions` / `normals` / `texCoords` / `indices` non-empty and `indices.size() % 3 == 0`
  - Asserts that `getTangents()` returns an accessor (the DamagedHelmet.gltf vendored here does NOT declare TANGENT, so the vector SHALL be empty; the loader SHALL NOT throw for this reason)
  - Asserts `baseColorTexture`, `metallicRoughnessTexture`, `normalTexture`, `occlusionTexture`, `emissiveTexture` are all non-empty (no specific filename assertion)
- `throws_on_missing_file`
  - Loading a non-existent path throws `std::runtime_error`
- `throws_on_corrupt_file`
  - Loading `assets/models/viking_room/viking_room.obj` (not a glTF file) throws `std::runtime_error`

The test SHALL NOT verify rendering output and SHALL NOT assert specific texture filename strings beyond non-empty / reasonable-suffix checks.

#### Scenario: All gltf loader tests pass

- **WHEN** running `test_gltf_loader`
- **THEN** all assertions SHALL pass
