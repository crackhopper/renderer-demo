## Purpose

Define the current mesh-loading contract for OBJ and related mesh ingestion paths used by the renderer.

## Requirements

### Requirement: OBJ mesh loading
The mesh loading system SHALL provide an `ObjLoader` class capable of loading Wavefront OBJ files.

#### Scenario: Load valid OBJ file
- **WHEN** `ObjLoader::load("model.obj")` is called on a valid OBJ file
- **THEN** the file SHALL be parsed by tinyobjloader and positions, normals, and texture coordinates SHALL be extracted

#### Scenario: OBJ file not found
- **WHEN** `load()` is called with a non-existent OBJ file
- **THEN** a `std::runtime_error` SHALL be thrown with an appropriate message

### Requirement: OBJ mesh data access
The ObjLoader SHALL provide access to loaded mesh data including vertices, normals, and indices.

#### Scenario: Get vertex positions
- **WHEN** `getPositions()` is called after loading
- **THEN** a vector of `Vec3f` vertex positions SHALL be returned

#### Scenario: Get vertex normals
- **WHEN** `getNormals()` is called after loading
- **THEN** a vector of `Vec3f` vertex normals SHALL be returned

#### Scenario: Get vertex texture coordinates
- **WHEN** `getTexCoords()` is called after loading
- **THEN** a vector of `Vec2f` texture coordinates SHALL be returned

#### Scenario: Get indices
- **WHEN** `getIndices()` is called after loading
- **THEN** a vector of unsigned integer indices SHALL be returned

### Requirement: GLTF mesh loading
The mesh loading system SHALL provide a `GLTFLoader` class capable of loading glTF 2.0 files.

#### Scenario: Load valid GLTF file
- **WHEN** `GLTFLoader::load("model.gltf")` is called on a valid glTF file
- **THEN** the file SHALL be parsed and mesh data SHALL be extracted

### Requirement: GLTF mesh data access
The GLTFLoader SHALL provide access to loaded mesh data with the same interface as ObjLoader.

#### Scenario: Get GLTF vertex positions
- **WHEN** `getPositions()` is called after loading a GLTF file
- **THEN** a vector of `Vec3f` vertex positions SHALL be returned

### Requirement: Model file path convention
Model test assets SHALL be located under `assets/models/` instead of the root-level `models/` directory. Any code or documentation referencing `models/viking_room.obj` SHALL be updated to reference `assets/models/viking_room/viking_room.obj`.

#### Scenario: Viking room model at new path
- **WHEN** loading the viking_room model
- **THEN** the file path SHALL resolve to `assets/models/viking_room/viking_room.obj`

#### Scenario: Old models directory does not exist
- **WHEN** checking the repository root
- **THEN** no `models/` directory SHALL exist at the root level
