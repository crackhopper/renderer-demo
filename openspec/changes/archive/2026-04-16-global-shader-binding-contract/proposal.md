## Why

The current material system identifies material-owned UBO bindings via a hardcoded name match on `"MaterialUBO"`. This single-point convention blocks support for non-`MaterialUBO`-named material parameter blocks, multiple material-owned buffers, and a generic material loader. The system needs a formal ownership contract: a small, fixed set of engine-reserved binding names (system-owned), with everything else defaulting to material-owned.

## What Changes

- Introduce a `constexpr` compile-time reserved binding name set: `CameraUBO`, `LightUBO`, `Bones`
- Add an ownership classification utility: given a binding name, return whether it is system-owned or material-owned
- **BREAKING**: `MaterialInstance` construction no longer matches on `"MaterialUBO"` as a special name; instead it collects all non-reserved `UniformBuffer` bindings as material-owned
- **BREAKING**: `SceneNode` validation replaces hardcoded name checks in `requiresRenderableOwnedResource()` with the formal reserved-name query
- Add validation: if a shader declares a reserved name with a type/layout inconsistent with the system contract, the system must treat it as an authoring error (FATAL)
- Update `MaterialParameterDataResource::getBindingName()` to return the actual reflected binding name instead of hardcoded `"MaterialUBO"`

## Capabilities

### New Capabilities
- `shader-binding-ownership`: Compile-time reserved binding name set and ownership classification utility; reserved-name misuse validation

### Modified Capabilities
- `material-system`: MaterialInstance construction uses ownership contract instead of `"MaterialUBO"` name match; `getDescriptorResources()` uses actual binding name; first-version still single material-owned UBO (multi-buffer deferred to REQ-032)
- `scene-node-validation`: `requiresRenderableOwnedResource()` replaced by formal ownership query against reserved-name set

## Impact

- `src/core/asset/material_instance.{hpp,cpp}` — constructor, `findMaterialUboBinding()`, `MaterialParameterDataResource::getBindingName()`
- `src/core/scene/object.cpp` — `requiresRenderableOwnedResource()` and validation logic
- New file: `src/core/asset/shader_binding_ownership.hpp` — reserved name set and query utility
- `notes/subsystems/material-system.md` — update ownership convention section
- `openspec/specs/material-system/spec.md` — update UBO allocation requirement
- `openspec/specs/scene-node-validation/spec.md` — update resource ownership validation
