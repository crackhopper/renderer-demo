## MODIFIED Requirements

### Requirement: MaterialInstance allocates UBO buffer from reflection
`MaterialInstance::create(template, passFlag)` SHALL walk the template shader's reflection bindings, locate non-system-owned `ShaderPropertyType::UniformBuffer` bindings using the ownership query from `shader-binding-ownership`, and allocate `m_uboBuffer` sized to the selected binding's `size`, zero-initialized. The binding pointer SHALL be cached as `m_uboBinding` (non-owning).

In the first version (before REQ-032 multi-buffer support), if exactly one non-system-owned `UniformBuffer` binding exists, it SHALL be selected. If multiple non-system-owned `UniformBuffer` bindings exist, the system SHALL assert in debug builds with a diagnostic message indicating that multi-buffer support is not yet available, and select the first one found. Shaders without any non-system-owned `UniformBuffer` binding MUST produce a `MaterialInstance` with an empty buffer and a null `m_uboBinding`; all setter calls on such an instance MUST assert in debug builds.

The system MUST NOT match on the name `"MaterialUBO"` as a special case. A shader that names its material uniform block `"SurfaceParams"` or any other non-reserved name SHALL be treated identically to one that names it `"MaterialUBO"`.

The `MaterialParameterDataResource` wrapper SHALL accept the actual reflected binding name at construction and SHALL return that name from `getBindingName()`. It MUST NOT return a hardcoded `StringID("MaterialUBO")`.

#### Scenario: Construction sizes buffer from reflection using ownership query
- **WHEN** `MaterialInstance::create(tmpl)` is called and the shader's reflection reports a non-system-owned `UniformBuffer` binding named `SurfaceParams` with `size = 48`
- **THEN** `m_uboBuffer.size() == 48` and every byte is zero, and `getDescriptorResources()` includes a resource whose `getBindingName()` returns `StringID("SurfaceParams")`

#### Scenario: MaterialUBO name still works as an ordinary material-owned binding
- **WHEN** a shader declares `uniform MaterialUBO { vec3 baseColor; float shininess; }` and reflection reports a `UniformBuffer` binding named `MaterialUBO`
- **THEN** the system treats it as a non-system-owned binding, allocates the buffer, and the material instance functions identically to before this change

#### Scenario: Shader without material-owned UBO produces empty buffer
- **WHEN** a shader's reflection bindings contain only `Texture2D` entries and system-owned `CameraUBO`
- **THEN** the resulting `MaterialInstance` has `m_uboBuffer.empty() == true` and `m_uboBinding == nullptr`

#### Scenario: Multiple non-system-owned UBOs assert in first version
- **WHEN** a shader declares two non-system-owned `UniformBuffer` bindings (`SurfaceParams` and `DetailParams`)
- **THEN** the system asserts in debug builds with a message about multi-buffer support, and selects the first binding found

#### Scenario: Setter type mismatch is an assertion failure
- **WHEN** `setFloat(StringID("baseColor"), 1.0f)` is called and reflection reports `baseColor` with type `Vec3`
- **THEN** an assertion fires in debug builds and the buffer is not modified

#### Scenario: Unknown member name asserts and is ignored
- **WHEN** `setVec4(StringID("doesNotExist"), ...)` is called and no member with that name exists in `m_uboBinding->members`
- **THEN** an assertion fires in debug builds and `m_uboBuffer` is unchanged
