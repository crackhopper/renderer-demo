## Why

`ShaderReflector` currently extracts only `(set, binding)` and total UBO byte size; it does not expose the name/type/offset of members inside a `UniformBuffer` struct. As a result, every shader-driven material (e.g. `BlinnPhongMaterialUBO`) must hand-write a std140-aligned C++ struct that is kept in lockstep with the GLSL declaration. This blocks REQ-005's unified `MaterialInstance` (which needs reflection-driven UBO writes) and REQ-003b's `PipelineBuildInfo.bindings` (which must replace hard-coded slot tables).

## What Changes

- Extend `ShaderResourceBinding` with a `std::vector<StructMemberInfo> members` field populated only for `UniformBuffer` bindings.
- Add `StructMemberInfo { name, type, offset, size }` to `src/core/resources/shader.hpp`.
- Add `Int` to `ShaderPropertyType` so integer UBO members (`enableAlbedo`, `enableNormalMap`, etc.) have a valid type.
- Extend `ShaderReflector::reflectSingleStage()` to walk spirv-cross struct member types and fill `members` using `get_member_name` / `DecorationOffset` / `get_declared_struct_member_size`.
- Preserve determinism by keeping `members` in spirv-cross's declared member order (no re-sorting).
- Merge guard: when `reflect()` merges the same `(set, binding)` across stages, `members` must be identical — take the first non-empty vector and assert equality.
- Nested struct / array-of-struct UBOs stay unsupported in this change; log and fall back to empty `members`.

## Capabilities

### New Capabilities
<!-- None -->

### Modified Capabilities
- `shader-reflection`: `ShaderResourceBinding` gains per-member UBO layout info; `ShaderReflector` walks struct members for uniform buffers.

## Impact

- **Code**:
  - `src/core/resources/shader.hpp` — new type + field + enum entry
  - `src/infra/shader_compiler/shader_reflector.cpp` — new `extractStructMembers` / `mapMemberType` helpers, call site in `extractBindings`, merge guard in `reflect()`
  - `src/test/integration/test_shader_compiler.cpp` — new assertions for `MaterialUBO` members
- **Specs**: `shader-reflection` spec gains member-extraction requirements
- **Downstream**: REQ-005 and REQ-003b are unblocked after this lands
- **Dependencies**: none — purely additive; existing callers that ignore `members` keep working
- **Breaking changes**: none
