## 1. MaterialTemplate Per-Pass Interface

- [x] 1.1 Add `getMaterialBindings(StringID pass)` to `MaterialTemplate` that returns material-owned bindings for a given pass
- [x] 1.2 Extend `buildBindingCache()` to populate per-pass material-owned binding lists using `isSystemOwnedBinding()`
- [x] 1.3 Add cross-pass same-name binding conflict detection (warn on inconsistent type/size/members)

## 2. MaterialInstance Multi-Buffer Slots

- [x] 2.1 Define `MaterialBufferSlot` struct (bindingName, binding pointer, byte buffer, IRenderResource, dirty flag)
- [x] 2.2 Replace `m_uboBuffer` / `m_uboBinding` / `m_uboResource` / `m_uboDirty` with `std::vector<MaterialBufferSlot> m_bufferSlots`
- [x] 2.3 Update constructor to create buffer slots from per-pass material-owned bindings; support both `UniformBuffer` and `StorageBuffer`
- [x] 2.4 Update `MaterialParameterDataResource` to accept `ResourceType` (UBO or StorageBuffer) at construction
- [x] 2.5 Add unsupported descriptor type fail-fast: FATAL on non-supported material-owned binding types during construction

## 3. New setParameter API

- [x] 3.1 Implement `setParameter(bindingName, memberName, value)` overloads (float, int32_t, Vec3f, Vec4f) with private `writeSlotMember` helper
- [x] 3.2 Refactor old convenience setters (`setFloat`, `setVec3`, `setVec4`, `setInt`) to search across slots; assert on ambiguous member name

## 4. Pass-Aware Descriptor Resources

- [x] 4.1 Change `getDescriptorResources()` to `getDescriptorResources(StringID pass)` using template's per-pass material bindings
- [x] 4.2 Update `syncGpuData()` to iterate all buffer slots
- [x] 4.3 Update `SceneNode::rebuildValidatedCache()` in `object.cpp` to call `getDescriptorResources(pass)`
- [x] 4.4 Update `RenderableSubMesh::getDescriptorResources()` (legacy) to forward with pass
- [x] 4.5 Update `SceneNode::getDescriptorResources()` to forward with pass
- [x] 4.6 Update `IRenderable::getDescriptorResources()` interface signature

## 5. Backward Compatibility Getters

- [x] 5.1 Add `getParameterBuffer(StringID bindingName)` and keep `getParameterBuffer()` as single-slot shortcut
- [x] 5.2 Add `getParameterBinding(StringID bindingName)` and keep `getParameterBinding()` as single-slot shortcut
- [x] 5.3 Update `getBufferSlotCount()` or equivalent accessor for test visibility

## 6. Tests

- [x] 6.1 Update existing tests to use `getDescriptorResources(Pass_Forward)` instead of `getDescriptorResources()`
- [x] 6.2 Add test: multi-buffer MaterialInstance with two UBOs, `setParameter` writes to each independently
- [x] 6.3 Add test: `getDescriptorResources(pass)` returns different resource sets for forward vs shadow
- [x] 6.4 Add test: convenience setter with single buffer still works
- [x] 6.5 Verify existing integration tests pass (blinnphong loader, scene node validation)

## 7. Documentation

- [x] 7.1 Update `notes/subsystems/material-system.md`
- [x] 7.2 Sync specs via `openspec` archive workflow
