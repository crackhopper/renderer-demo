## Context

`ShaderReflector` (in `src/infra/shader_compiler/shader_reflector.cpp`) wraps spirv-cross and produces `std::vector<ShaderResourceBinding>`. For `UniformBuffer` bindings it currently calls `compiler.get_declared_struct_size(type)` to fill `size`, but does not walk the struct's members. This forces every material that writes to a UBO to hand-write an aligned C++ struct (e.g. `BlinnPhongMaterialUBO` in `material.hpp`) and keep it manually in sync with the GLSL declaration.

REQ-005 (unified material) and REQ-003b (pipeline prebuilding) both need per-member layout metadata:
- REQ-005: `MaterialInstance::setVec4(StringID, Vec4f)` must look up the member's std140 offset from the binding and `memcpy` into a raw byte buffer.
- REQ-003b: `PipelineBuildInfo.bindings` replaces `PipelineSlotDetails`; a driver-friendly `ShaderResourceBinding` must carry enough layout info that backends can construct descriptor set layouts without additional lookup tables.

This change is the minimum-viable reflection extension that unblocks both.

## Goals / Non-Goals

**Goals:**
- Fully describe the layout of each `UniformBuffer` binding: per-member name, type, std140 offset, std140 size.
- Keep the API purely additive — existing callers that ignore `members` continue to compile and run.
- Preserve `(set, binding)` merging across stages; members must be consistent and safely deduplicated.
- Add a single integer-typed enum value (`ShaderPropertyType::Int`) so UBO `int` fields have a valid type tag.

**Non-Goals:**
- Nested struct UBOs and arrays of structs (`member[N].field`) are out of scope — flag them with a log and fall back to empty `members`.
- Exposing spirv-cross types in `core` headers; all translation happens inside `infra/shader_compiler`.
- Push-constant block member reflection (separate concern, handled if/when push constants become data-driven).
- Changing the existing `ShaderResourceBinding::operator==` — members are derived from `(set, binding)` and don't need to participate in equality.

## Decisions

### Decision 1: Add `StructMemberInfo` rather than a map keyed by name
- **Choice**: Store `std::vector<StructMemberInfo>` with `{name, type, offset, size}` per entry. Downstream lookups (REQ-005) can build a `unordered_map<StringID, offset>` locally if needed.
- **Alternative considered**: A map keyed by member name inside `ShaderResourceBinding`.
- **Rationale**: A vector keeps spirv-cross's declared order stable (useful for debug printing and deterministic hashing) and avoids paying for a hash table when member counts are small (< 16 typically). Consumers that need O(1) name lookup can cache locally.

### Decision 2: Introduce a separate `mapMemberType(const SPIRType&)` helper
- **Choice**: Keep the existing `mapSpvType(compiler, res, storageClass)` for resource-level mapping and add `mapMemberType(type)` for struct-member types (float/int/vec2/vec3/vec4/mat4).
- **Alternative considered**: Extend `mapSpvType` to handle both cases.
- **Rationale**: The two cases are structurally different — resource mapping uses `Resource`, member mapping uses a bare `SPIRType` read from `type.member_types[i]`. Mixing them into one function would require overloaded parameters and branching on whether we have a `Resource` or just a `Type`, which would be harder to read than two small helpers.

### Decision 3: Merge guard uses first-non-empty + debug assert
- **Choice**: When `reflect()` merges the same `(set, binding)` across stages, the first stage's `members` wins. An assert compares subsequent stages' `members` for equality in debug builds.
- **Alternative considered**: Always overwrite with the latest stage, or require explicit ordering.
- **Rationale**: Same UBO declared in both vert and frag must be structurally identical (compiler enforces this). First-wins is simplest and any divergence indicates a source bug that the assert catches in debug builds without adding runtime cost in release.

### Decision 4: Fall back silently on unsupported shapes
- **Choice**: Nested structs and arrays-of-struct populate an empty `members` vector and emit a log line. The binding itself is still returned with correct `size`.
- **Alternative considered**: Throw or return `std::nullopt`.
- **Rationale**: Existing BlinnPhong UBO is flat, so the feature can land without supporting everything. Failing hard would block shaders that currently work; silent fallback keeps the old path functional until nested support is added.

### Decision 5: Add `ShaderPropertyType::Int`
- **Choice**: Add a single new enum value rather than shoe-horning integers into `Float`.
- **Alternative considered**: Keep the enum unchanged and use `Float` for int members (since they're 4 bytes in std140).
- **Rationale**: REQ-005 will assert `m.type == ShaderPropertyType::Vec4` inside `setVec4`. If `int` and `float` both map to `Float`, `setInt` cannot safely assert against the wrong setter. Correct typing at the reflection layer is cheap and prevents misuse downstream.

## Risks / Trade-offs

- **[Risk]** spirv-cross member-name retrieval (`get_member_name`) can return empty strings for stripped SPIR-V. → **Mitigation**: If `members[i].name.empty()` after extraction, synthesize `"_member" + std::to_string(i)` so downstream lookups by `StringID` still work deterministically, and log a warning.
- **[Risk]** The `members` vector increases `ShaderResourceBinding` size; some code copies these bindings. → **Mitigation**: Typical UBO has 5-10 members, each ~40 bytes → ~400 bytes per binding. Acceptable given binding count is tens per shader. Profile after landing.
- **[Risk]** Existing reflection tests are loose — a regression might not be caught. → **Mitigation**: Add explicit assertions in `test_shader_compiler.cpp` (see proposal).
- **[Trade-off]** Declared order is preserved, not sorted. Any downstream hash that cares about stability must either sort locally or trust spirv-cross ordering. Explicitly documented in the spec requirement.

## Migration Plan

Pure addition — no migration needed.

- Existing callers that only read `(set, binding, type, size)` remain correct; the new `members` vector is unused and empty for non-UBO bindings.
- `BlinnPhongMaterialUBO` and `DrawMaterial` are untouched by this change; REQ-005 will remove them after this lands.
- Rollback: revert the commit; `ShaderResourceBinding` returns to pre-change form.

## Open Questions

None at this time. Nested struct handling is deferred to a future change when a shader actually requires it.
