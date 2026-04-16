## Context

The current material system identifies its owned UBO via `findMaterialUboBinding()` which matches `binding.name == "MaterialUBO"`. Similarly, `requiresRenderableOwnedResource()` in `object.cpp` uses hardcoded name checks to classify bindings as scene-owned or renderable-owned. This approach blocks REQ-032 (multi-buffer material interface) and REQ-033 (generic material loader).

Current ownership logic (in `object.cpp:76-84`):
- `CameraUBO`, `LightUBO` → scene-owned (not renderable-owned)
- `MaterialUBO` → renderable-owned (material provides it)
- `Bones` → renderable-owned (skeleton provides it)
- Everything else → returns `false` (silently ignored)

The "everything else returns false" is the key gap: non-reserved, non-`MaterialUBO` bindings are ignored rather than treated as material-owned.

## Goals / Non-Goals

**Goals:**
- Define a formal `constexpr` set of engine-reserved (system-owned) binding names
- Provide a single-function ownership query: `isSystemOwnedBinding(name) -> bool`
- Replace `findMaterialUboBinding()` with "find all non-reserved UniformBuffer bindings"
- Replace hardcoded name checks in `requiresRenderableOwnedResource()` with the ownership query
- Validate reserved-name misuse at material/node construction time

**Non-Goals:**
- Multi-buffer material support (REQ-032)
- Pass-aware descriptor resource collection (REQ-032)
- YAML material asset format (REQ-033)
- Changing how CameraUBO/LightUBO/Bones are actually provided (scene vs skeleton)

## Decisions

### D1: constexpr array + free function

**Choice**: A header-only `shader_binding_ownership.hpp` with a `constexpr std::array` of reserved names and an `inline constexpr` query function.

**Alternatives considered**:
- Runtime registry: unnecessary complexity; REQ-031 R5 says expansion requires new requirement
- Enum-based: names come from shader reflection as strings, enum adds a mapping layer with no benefit

**Form**:
```cpp
// src/core/asset/shader_binding_ownership.hpp
namespace LX_core {
inline constexpr std::string_view kSystemOwnedBindings[] = {
    "CameraUBO", "LightUBO", "Bones"
};
inline bool isSystemOwnedBinding(std::string_view name) {
    for (auto sv : kSystemOwnedBindings)
        if (sv == name) return true;
    return false;
}
}
```

### D2: MaterialInstance construction — first-version bridge

This change keeps MaterialInstance with a single UBO buffer (matching current behavior), but the selection logic changes from "find binding named MaterialUBO" to "find the first non-reserved UniformBuffer binding". This is a stepping stone; REQ-032 will generalize to multi-buffer.

For first-version, if multiple non-reserved UBO bindings exist, assert and pick the first one (REQ-032 will handle the multi-buffer case properly).

### D3: MaterialParameterDataResource stores actual binding name

`MaterialParameterDataResource::getBindingName()` currently returns a hardcoded `StringID("MaterialUBO")`. Change it to accept the actual binding name at construction and return that. This is needed so `SceneNode` validation can match resources to bindings by name without assuming `MaterialUBO`.

### D4: requiresRenderableOwnedResource refactored

Replace the current function with ownership-based logic:
- System-owned bindings (`isSystemOwnedBinding`) → not renderable-owned (scene provides them), **except** `Bones` which is renderable-owned (provided by Skeleton on the node)
- Non-system-owned bindings → renderable-owned (material provides them)

This preserves the semantic distinction: `CameraUBO`/`LightUBO` come from scene, `Bones` comes from skeleton on the node, everything else comes from material.

### D5: Reserved-name misuse validation

If a shader declares a reserved name with a type that doesn't match the system contract:
- `CameraUBO` must be `UniformBuffer`
- `LightUBO` must be `UniformBuffer`
- `Bones` must be `StorageBuffer` or `UniformBuffer` (currently UBO)

Validation happens during `SceneNode::rebuildValidatedCache()` alongside existing structural checks. Misuse is FATAL + terminate, consistent with existing validation philosophy.

### D6: SceneNode validation — material-owned resource check generalization

Current code checks specifically for `"MaterialUBO"` presence. Replace with: for each non-reserved binding in the shader reflection, verify the material's descriptor resources contain a resource with a matching binding name. This generalizes the check to work with any material-owned binding name.

## Risks / Trade-offs

- **[Risk]** First-version single-UBO bridge may be confusing alongside the "any non-reserved UBO" selection.
  → Mitigation: clear assert message if multiple non-reserved UBOs found; REQ-032 follows immediately.

- **[Risk]** Existing shaders that use `MaterialUBO` as a name continue to work, but the name is no longer special.
  → Mitigation: this is the desired outcome per REQ-031 R9. No shader changes needed.

- **[Risk]** `Bones` is reserved but still renderable-owned (skeleton provides it, not scene).
  → Mitigation: "system-owned" means "not material-owned" in the ownership contract. The provider (scene vs skeleton) is orthogonal and remains unchanged.
