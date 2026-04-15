## ADDED Requirements

### Requirement: ShaderReflector extracts vertex-stage input attributes
ShaderReflector SHALL parse SPIR-V vertex-stage input declarations and expose them as reflection data alongside descriptor bindings. Each reflected vertex input attribute MUST capture at minimum the declared name, numeric location, and a type tag sufficient to compare against mesh `VertexLayout` semantics during structural validation.

The reflection output MUST be deterministic for identical SPIR-V input and MUST preserve declaration-order-independent matching by location.

#### Scenario: Vertex inputs reflected from a skinned shader
- **WHEN** a vertex shader declares inputs for position, normal, blend weights, and blend indices
- **THEN** the reflection result includes all four attributes with their declared locations and comparable type metadata

#### Scenario: Reflection is stable across repeated runs
- **WHEN** the same vertex SPIR-V is reflected twice
- **THEN** the vertex input attribute set is identical across both runs

### Requirement: CompiledShader exposes reflected vertex input contract
`CompiledShader` SHALL expose read access to the reflected vertex-stage input contract so that higher layers can validate mesh vertex layout compatibility without re-running SPIR-V reflection. This contract SHALL correspond to the exact shader variant combination compiled for the owning material pass.

#### Scenario: Variant-specific vertex contract is observable
- **WHEN** two compiled shader variants differ in whether skinning inputs are enabled
- **THEN** querying their reflected vertex input contracts shows the skinned variant requiring the additional skinning attributes while the non-skinned variant does not
