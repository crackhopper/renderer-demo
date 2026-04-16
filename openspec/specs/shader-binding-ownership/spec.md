# shader-binding-ownership Specification

## Purpose
TBD - created by archiving change global-shader-binding-contract. Update Purpose after archive.
## Requirements
### Requirement: Engine provides a compile-time reserved binding name set
The system SHALL define a compile-time constant array of engine-reserved (system-owned) binding names. The first version of this set SHALL contain exactly three names: `CameraUBO`, `LightUBO`, and `Bones`. No other names SHALL be included in the reserved set without a new requirement or spec change.

The reserved set SHALL be defined in a dedicated header (`shader_binding_ownership.hpp`) within the core asset layer and SHALL be usable at compile time.

#### Scenario: Reserved set contains exactly three names
- **WHEN** the reserved binding name set is inspected
- **THEN** it contains exactly `CameraUBO`, `LightUBO`, and `Bones`, and no other entries

#### Scenario: Reserved set is usable at compile time
- **WHEN** a `constexpr` or `static_assert` context references the reserved set
- **THEN** the code compiles without error

### Requirement: Ownership query classifies bindings as system-owned or material-owned
The system SHALL provide a function `isSystemOwnedBinding(name)` that returns `true` if and only if the given name is in the reserved binding name set. All bindings not in the reserved set SHALL be classified as material-owned by default.

This function SHALL be the single authoritative source for ownership classification. No other code path SHALL use hardcoded name comparisons to determine whether a binding belongs to the system or to the material.

#### Scenario: Reserved name is classified as system-owned
- **WHEN** `isSystemOwnedBinding("CameraUBO")` is called
- **THEN** the function returns `true`

#### Scenario: Non-reserved name is classified as material-owned
- **WHEN** `isSystemOwnedBinding("SurfaceParams")` is called
- **THEN** the function returns `false`

#### Scenario: MaterialUBO is not system-owned
- **WHEN** `isSystemOwnedBinding("MaterialUBO")` is called
- **THEN** the function returns `false`, because `MaterialUBO` is not in the reserved set and is treated as an ordinary material-owned binding

### Requirement: Reserved binding name misuse is a fatal authoring error
If a shader declares a binding with a reserved name but the binding's descriptor type is inconsistent with the system contract, the system SHALL treat this as a shader authoring error. The system MUST emit a `FATAL` log with diagnostic context (binding name, expected type, actual type) and terminate the process immediately.

The system contract for reserved names in the first version SHALL be:
- `CameraUBO`: `UniformBuffer`
- `LightUBO`: `UniformBuffer`
- `Bones`: `UniformBuffer`

#### Scenario: CameraUBO declared as Texture2D is a fatal error
- **WHEN** a shader's reflection reports a binding named `CameraUBO` with type `Texture2D`
- **THEN** the system emits a `FATAL` log identifying the type mismatch and terminates immediately

#### Scenario: CameraUBO declared as UniformBuffer passes validation
- **WHEN** a shader's reflection reports a binding named `CameraUBO` with type `UniformBuffer`
- **THEN** the system accepts the binding without error

#### Scenario: Bones declared as StorageBuffer is a fatal error
- **WHEN** a shader's reflection reports a binding named `Bones` with type `StorageBuffer`
- **THEN** the system emits a `FATAL` log identifying the type mismatch and terminates immediately

