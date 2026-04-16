## Why

The current `MaterialInstance` has three structural limitations blocking generality:

1. **Single UBO buffer**: only one `m_uboBuffer` / `m_uboBinding` — multiple material-owned buffers cannot be expressed.
2. **Flat binding cache**: `MaterialTemplate::m_bindingCache` merges all pass bindings into one name-keyed map, losing pass scope. Same-name bindings across passes silently overwrite.
3. **Member-only addressing**: `setFloat(memberName, value)` works only when there's a single buffer. With multiple buffers, the binding name is required to disambiguate.

Additionally, `getDescriptorResources()` is not pass-aware — it returns all material resources regardless of which pass is being rendered.

REQ-031 (now complete) established the ownership contract. This change builds on it to create the full pass-aware material binding interface.

## What Changes

- **BREAKING**: `MaterialInstance` replaces single `m_uboBuffer` / `m_uboBinding` / `m_uboResource` with a `map<StringID, MaterialBufferSlot>` keyed by binding name
- **BREAKING**: `getDescriptorResources()` becomes `getDescriptorResources(StringID pass)` — pass-aware
- **BREAKING**: New primary parameter API: `setParameter(bindingName, memberName, value)` for buffer writes
- Old convenience setters (`setFloat`, `setVec3`, etc.) remain as single-buffer shortcuts that assert if multiple buffers exist
- `MaterialTemplate` builds per-pass material-owned binding interface alongside the existing flattened cache
- Multi-pass same-name binding conflict detection: inconsistent layouts across passes are detected and reported
- First-version supported descriptor types: `UniformBuffer`, `StorageBuffer`, `Texture2D`, `TextureCube`
- Unsupported descriptor types (`Sampler`, `StorageImage`, etc.) fail fast at interface construction

## Capabilities

### New Capabilities
_None — all changes are modifications to existing capabilities._

### Modified Capabilities
- `material-system`: Multi-buffer MaterialInstance, pass-aware getDescriptorResources, new setParameter API, per-pass material interface in template, supported descriptor type enforcement, cross-pass binding conflict detection

## Impact

- `src/core/asset/material_template.hpp` — per-pass material-owned binding interface
- `src/core/asset/material_instance.{hpp,cpp}` — multi-buffer slots, new API, pass-aware descriptor collection
- `src/core/asset/material_pass_definition.hpp` — material-owned bindings per pass
- `src/core/scene/object.cpp` — adapt to pass-aware `getDescriptorResources(pass)`
- `src/infra/material_loader/blinn_phong_material_loader.cpp` — no API changes needed (single buffer, old setters still work)
- `src/backend/vulkan/details/commands/command_buffer.*` — adapt descriptor bind path
- `notes/subsystems/material-system.md` — update design doc
