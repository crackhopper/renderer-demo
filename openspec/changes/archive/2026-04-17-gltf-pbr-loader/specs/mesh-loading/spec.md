## MODIFIED Requirements

### Requirement: GLTF mesh loading

The mesh loading system SHALL provide a `GLTFLoader` class capable of loading glTF 2.0 files. Parsing SHALL be implemented by the vendored `cgltf` library (see capability `gltf-pbr-loader`), not by any stub. The loader SHALL support ASCII `.gltf` with external `.bin` and external image references as its primary path; `.glb` MAY also work via cgltf's own auto-detection but is not a primary acceptance target. The loader SHALL throw `std::runtime_error` on parse failure with a message that contains the input filename.

#### Scenario: Load valid GLTF file

- **WHEN** `GLTFLoader::load("model.gltf")` is called on a valid glTF 2.0 file whose `meshes[0].primitives[0]` is triangles and has POSITION
- **THEN** the file SHALL be parsed via `cgltf_parse_file` + `cgltf_load_buffers` and geometry + PBR metadata SHALL be extracted from `meshes[0].primitives[0]`

#### Scenario: Load stub is removed

- **WHEN** `GLTFLoader::load` is called on a valid `.gltf` file
- **THEN** the call SHALL NOT throw `"ASCII GLTF (.gltf) not yet supported"` or `"Binary GLTF (.glb) not yet supported"` — those stub messages SHALL NOT appear in the implementation

### Requirement: GLTF mesh data access

The `GLTFLoader` SHALL provide access to loaded mesh data, including the same geometry interface as `ObjLoader` plus glTF-specific PBR extensions.

The public header `src/infra/mesh_loader/gltf_mesh_loader.hpp` SHALL declare, in addition to the position/normal/texCoord/indices accessors:

```cpp
const std::vector<LX_core::Vec4f>& getTangents() const;
const GLTFPbrMaterial&             getMaterial() const;
```

`getTangents()` SHALL return an empty vector when the glTF file does not declare a `TANGENT` accessor for the consumed primitive; the loader SHALL NOT generate tangents automatically. `getMaterial()` SHALL return a `GLTFPbrMaterial` describing the primitive's bound glTF material (see capability `gltf-pbr-loader`).

#### Scenario: Get GLTF vertex positions

- **WHEN** `getPositions()` is called after loading a glTF file
- **THEN** a vector of `Vec3f` vertex positions SHALL be returned

#### Scenario: Get tangents when present

- **WHEN** `getTangents()` is called after loading a glTF file that declares a `TANGENT` accessor
- **THEN** a non-empty vector of `Vec4f` tangents SHALL be returned

#### Scenario: Get tangents when absent

- **WHEN** `getTangents()` is called after loading a glTF file with no `TANGENT` accessor
- **THEN** an empty vector SHALL be returned and no tangent generation SHALL have occurred

#### Scenario: Get PBR material metadata

- **WHEN** `getMaterial()` is called after loading
- **THEN** a `GLTFPbrMaterial` SHALL be returned whose factors and texture URI strings reflect the primitive's glTF material (or struct defaults if no material is bound)
