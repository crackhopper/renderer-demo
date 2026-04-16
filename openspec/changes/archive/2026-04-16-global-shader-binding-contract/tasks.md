## 1. Ownership Utility

- [x] 1.1 Create `src/core/asset/shader_binding_ownership.hpp` with `kSystemOwnedBindings` constexpr array and `isSystemOwnedBinding()` query function
- [x] 1.2 Add reserved-name type contract: `getExpectedTypeForSystemBinding(name)` returning the expected `ShaderPropertyType` for each reserved name

## 2. MaterialInstance Ownership Migration

- [x] 2.1 Replace `findMaterialUboBinding()` in `material_instance.cpp` with a new helper that finds the first non-system-owned `UniformBuffer` binding using `isSystemOwnedBinding()`; assert if multiple found
- [x] 2.2 Update `MaterialParameterDataResource` to accept binding name at construction; remove hardcoded `StringID("MaterialUBO")` from `getBindingName()`
- [x] 2.3 Update `MaterialInstance` constructor to pass the actual binding name to `MaterialParameterDataResource`

## 3. SceneNode Validation Migration

- [x] 3.1 Replace `requiresRenderableOwnedResource()` in `object.cpp` with ownership-query-based logic using `isSystemOwnedBinding()`
- [x] 3.2 Replace the `hasMaterialUbo` check (matching `StringID("MaterialUBO")`) with a generic non-reserved binding presence check by `getBindingName()`
- [x] 3.3 Add reserved-name type validation: for each system-owned binding in shader reflection, verify its type matches the system contract; FATAL on mismatch

## 4. Tests

- [x] 4.1 Add unit test for `isSystemOwnedBinding()`: reserved names return true, non-reserved return false, `MaterialUBO` returns false
- [x] 4.2 Update or add integration test: `MaterialInstance` constructed with a shader using `SurfaceParams` (non-`MaterialUBO` name) works correctly
- [x] 4.3 Verify existing integration tests still pass (blinnphong material uses `MaterialUBO` name and should work unchanged)

## 5. Documentation

- [x] 5.1 Update `notes/subsystems/material-system.md` ownership convention section
- [x] 5.2 Sync specs via `openspec` archive workflow
