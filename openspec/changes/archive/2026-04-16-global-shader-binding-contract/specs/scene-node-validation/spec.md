## MODIFIED Requirements

### Requirement: SceneNode validates forward-shader resource requirements from variants
For any enabled pass backed by the `blinnphong_0` forward shader family, `SceneNode` SHALL validate mesh and node resources against the active shader variant set before the node is considered structurally valid.

At minimum, the validation rules SHALL be:

- `USE_VERTEX_COLOR => mesh` provides `inColor`
- `USE_UV => mesh` provides `inUV`
- `USE_LIGHTING => mesh` provides `inNormal`
- `USE_NORMAL_MAP => mesh` provides `inTangent` and `inUV`
- `USE_SKINNING => mesh` provides `inBoneIDs` and `inBoneWeights`, and the node provides `Skeleton/Bones`

Additionally, `SceneNode` SHALL validate descriptor resource ownership using the `isSystemOwnedBinding()` query from `shader-binding-ownership`. The validation logic MUST NOT use hardcoded binding name comparisons (such as checking for `"MaterialUBO"`, `"CameraUBO"`, or `"LightUBO"` by literal string). Instead:

- For each binding in the shader reflection:
  - If `isSystemOwnedBinding(binding.name)` returns `true` and the binding is `Bones`: validate that the node provides a `Skeleton` with a `Bones` UBO resource.
  - If `isSystemOwnedBinding(binding.name)` returns `true` and the binding is not `Bones`: skip (scene provides these resources, not the renderable).
  - If `isSystemOwnedBinding(binding.name)` returns `false`: validate that the material's descriptor resources contain a resource with a matching `getBindingName()`.

Reserved binding name misuse (type mismatch against the system contract) SHALL be validated as specified by `shader-binding-ownership` and SHALL be treated as `FATAL + terminate`.

Any mismatch between the enabled variant set and the available mesh/skeleton/material resources MUST be treated as a structural validation failure and handled as `FATAL + terminate`.

#### Scenario: Missing vertex color attribute terminates
- **WHEN** a pass enables `USE_VERTEX_COLOR` and the mesh vertex layout does not provide `inColor`
- **THEN** `SceneNode` logs a `FATAL` structural validation failure for that pass and terminates immediately

#### Scenario: Missing UV for textured forward pass terminates
- **WHEN** a pass enables `USE_UV` and the mesh vertex layout does not provide `inUV`
- **THEN** `SceneNode` logs a `FATAL` structural validation failure for that pass and terminates immediately

#### Scenario: Missing tangent for normal-mapped pass terminates
- **WHEN** a pass enables `USE_NORMAL_MAP` and the mesh vertex layout does not provide `inTangent`
- **THEN** `SceneNode` logs a `FATAL` structural validation failure for that pass and terminates immediately

#### Scenario: Missing skeleton resources for skinned pass terminates
- **WHEN** a pass enables `USE_SKINNING` and the node lacks a `Skeleton` or `Bones` resource
- **THEN** `SceneNode` logs a `FATAL` structural validation failure for that pass and terminates immediately

#### Scenario: Non-reserved material binding validated by name
- **WHEN** a shader declares a non-reserved `UniformBuffer` binding named `SurfaceParams` and the material's descriptor resources include a resource with `getBindingName() == StringID("SurfaceParams")`
- **THEN** the validation passes for that binding

#### Scenario: Missing non-reserved material binding terminates
- **WHEN** a shader declares a non-reserved `UniformBuffer` binding named `SurfaceParams` and the material's descriptor resources do not include a resource with that binding name
- **THEN** `SceneNode` logs a `FATAL` structural validation failure and terminates immediately

#### Scenario: Reserved name type mismatch terminates
- **WHEN** a shader declares `CameraUBO` as `Texture2D` instead of `UniformBuffer`
- **THEN** the system logs a `FATAL` error with type mismatch details and terminates immediately
