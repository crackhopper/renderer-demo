## 1. Core type additions

- [x] 1.1 Add `ShaderPropertyType::Int` to the enum in `src/core/resources/shader.hpp`
- [x] 1.2 Add `StructMemberInfo { std::string name; ShaderPropertyType type; uint32_t offset; uint32_t size; }` to `src/core/resources/shader.hpp`
- [x] 1.3 Add `std::vector<StructMemberInfo> members;` field to `ShaderResourceBinding` (after existing fields, before `operator==`)
- [x] 1.4 Confirm `ShaderResourceBinding::operator==` is unchanged (members not part of equality)

## 2. Reflector — helper functions

- [x] 2.1 Add `mapMemberType(const spirv_cross::SPIRType&)` helper in `src/infra/shader_compiler/shader_reflector.cpp` that handles `Float`/`Int` base types combined with `vecsize`/`columns` to return `Float`/`Int`/`Vec2`/`Vec3`/`Vec4`/`Mat4`
- [x] 2.2 Add `extractStructMembers(compiler, type, out)` that walks `type.member_types`, populates name (falling back to `"_memberN"` when empty), type, offset (`get_member_decoration(self, i, DecorationOffset)`), and size (`get_declared_struct_member_size`)
- [x] 2.3 Detect nested struct / array-of-struct members — if encountered, clear `out` and log a warning, then return early

## 3. Reflector — integration

- [x] 3.1 In `extractBindings` lambda, for resources extracted from `uniform_buffers` (storageClass == Uniform), after filling `size`, call `extractStructMembers(compiler, type, b.members)`
- [x] 3.2 In `ShaderReflector::reflect()` cross-stage merge, when merging an existing binding with a new one at the same `(set, binding)`, preserve the first non-empty `members` vector and debug-assert structural equality with any subsequent stage

## 4. Tests

- [x] 4.1 In `src/test/integration/test_shader_compiler.cpp`, add a test case `reflects_blinnphong_material_ubo_members` that compiles the existing `blinnphong_0.frag` (which declares `MaterialUBO`)
- [x] 4.2 Assert `binding.type == UniformBuffer` and `binding.members.size() >= 5`
- [x] 4.3 Assert that `baseColor` member exists with `type == Vec3` and `offset == 0`
- [x] 4.4 Assert that `shininess` member exists with `type == Float` and `offset == 16` — **observed offset is 12** (std140 packs `vec3 + float` into a 16-byte bucket); test accepts either 12 or 16 and actual value asserted is 12
- [x] 4.5 Assert that `enableAlbedo` member exists with `type == Int`
- [x] 4.6 Add a negative test: reflect a shader with only `sampler2D` bindings and assert that those bindings have empty `members`

## 5. Build verification

- [x] 5.1 Run `cmake --build build` — must succeed with no new warnings
- [x] 5.2 Run `test_shader_compiler` — must pass including new cases
- [x] 5.3 Run full test suite to confirm no regression in callers that ignore `members`
