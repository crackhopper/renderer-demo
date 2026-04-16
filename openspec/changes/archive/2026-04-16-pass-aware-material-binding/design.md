## Context

After REQ-031, the ownership contract is established: `isSystemOwnedBinding()` classifies bindings, and `MaterialInstance` finds its UBO by selecting non-system-owned `UniformBuffer` bindings. However, the current implementation still has:

- A single `m_uboBuffer` / `m_uboBinding` / `m_uboResource` (asserts on multiple UBOs)
- A template-level flattened `m_bindingCache` that loses pass scope
- `getDescriptorResources()` without a pass parameter
- `setFloat(memberName, value)` addressing that can't disambiguate across multiple buffers

The backend already consumes descriptor resources per-pass via `ValidatedRenderablePassData.descriptorResources`, so the downstream path is already pass-aware. The gap is in MaterialInstance and MaterialTemplate.

## Goals / Non-Goals

**Goals:**
- MaterialInstance supports N material-owned buffer slots (keyed by binding name)
- Each buffer slot has independent byte buffer, dirty state, and IRenderResource wrapper
- New `setParameter(bindingName, memberName, value)` API for buffer writes
- `getDescriptorResources(pass)` returns resources scoped to a target pass's reflection
- MaterialTemplate builds per-pass material-owned binding lists (alongside existing flattened cache)
- Cross-pass same-name binding conflict detection
- First-version supported types: UniformBuffer, StorageBuffer, Texture2D, TextureCube
- Unsupported types fail fast at template/interface construction

**Non-Goals:**
- YAML material asset format (REQ-033)
- Editor UI schema
- Changing the backend descriptor bind path (it already works by name)
- Removing the flattened binding cache entirely (still useful for texture lookup)

## Decisions

### D1: MaterialBufferSlot struct

A new `MaterialBufferSlot` struct holds per-buffer state:

```cpp
struct MaterialBufferSlot {
  StringID bindingName;
  const ShaderResourceBinding *binding; // non-owning, from reflection
  std::vector<uint8_t> buffer;
  IRenderResourcePtr resource; // MaterialParameterDataResource wrapper
  bool dirty = false;
};
```

`MaterialInstance` replaces `m_uboBuffer` / `m_uboBinding` / `m_uboResource` / `m_uboDirty` with:
```cpp
std::vector<MaterialBufferSlot> m_bufferSlots;
```

Vector is used (not map) because the count is small (typically 1-2) and iteration order matters for deterministic descriptor output.

### D2: setParameter(bindingName, memberName, value) as primary API

New primary API:
```cpp
void setParameter(StringID bindingName, StringID memberName, const void* data, size_t size, ShaderPropertyType expected);
// Type-safe wrappers:
void setParameter(StringID bindingName, StringID memberName, float value);
void setParameter(StringID bindingName, StringID memberName, const Vec3f& value);
// etc.
```

Old convenience setters (`setFloat`, `setVec3`, `setVec4`, `setInt`) are retained. They search across buffer slots by member name:
- If exactly one buffer slot contains a member with that name, write to it.
- If multiple slots contain a member with the same name, assert with diagnostic.
- This preserves backward compat for the common single-buffer case.

### D3: getDescriptorResources(pass) — pass-aware

`getDescriptorResources(StringID pass)` resolves resources for a specific pass:

1. Get the pass's shader reflection bindings
2. For each non-system-owned binding:
   - If it's a buffer type: find it in `m_bufferSlots` by name
   - If it's a texture type: find it in `m_textures` by name
3. Return in `(set << 16 | binding)` order for that pass

The old no-arg `getDescriptorResources()` is removed. All callers already have a pass context (SceneNode validation is per-pass, RenderingItem is per-pass).

### D4: MaterialTemplate per-pass material interface

`MaterialTemplate` adds a per-pass material-owned binding list built during `buildBindingCache()`:

```cpp
// For each pass, the list of material-owned bindings from that pass's shader reflection
std::unordered_map<StringID, std::vector<const ShaderResourceBinding*>, StringID::Hash> m_passMaterialBindings;
```

This is built alongside the existing flattened `m_bindingCache`. The flattened cache remains for backward-compat texture lookup (`findBinding` by name without pass).

### D5: Cross-pass same-name binding conflict detection

During `buildBindingCache()`, if two passes have a material-owned binding with the same name, verify:
- Same descriptor type
- Same buffer size (for buffer types)
- Same member layout (for UBOs)

If inconsistent, emit a warning log but do NOT fail — the pass-aware query will always resolve to the correct pass's binding. The data is stored per-pass, so no data is lost.

### D6: Supported descriptor type enforcement

During MaterialInstance construction, when iterating material-owned bindings:
- `UniformBuffer` → create a buffer slot
- `StorageBuffer` → create a buffer slot (same mechanism)
- `Texture2D` / `TextureCube` → handled by existing texture map
- Anything else (`Sampler`, etc.) → FATAL with diagnostic

### D7: StorageBuffer handling

`StorageBuffer` bindings use the same `MaterialBufferSlot` mechanism as `UniformBuffer`. The `MaterialParameterDataResource` already supports `ResourceType::UniformBuffer`; we add `ResourceType::StorageBuffer` support by letting the resource type be determined from the binding. The `StorageBuffer` wrapper returns `ResourceType::StorageBuffer` from `getType()`.

### D8: Backward compatibility for SceneNode / RenderableSubMesh

`SceneNode::rebuildValidatedCache()` currently calls `m_materialInstance->getDescriptorResources()`. It will be updated to call `getDescriptorResources(pass)` since validation is already per-pass.

`RenderableSubMesh::getDescriptorResources()` (legacy) will forward with `Pass_Forward` as a default for backward compat.

## Risks / Trade-offs

- **[Risk]** Breaking API change: `getDescriptorResources()` → `getDescriptorResources(pass)`.
  → Mitigation: All call sites are internal and already operate in a per-pass context. Mechanical update.

- **[Risk]** Old setters may mask bugs when multiple buffers have same-named members.
  → Mitigation: Assert on ambiguous member name. Callers should migrate to `setParameter()`.

- **[Risk]** StorageBuffer write API is raw byte buffer with no member-level access in first version.
  → Mitigation: Acceptable for first version. Member-level access for storage buffers can follow.
