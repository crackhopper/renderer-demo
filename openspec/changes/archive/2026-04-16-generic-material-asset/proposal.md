## Why

Every new material template currently requires a bespoke C++ loader (like `blinn_phong_material_loader.cpp`). Default parameter values, default textures, and per-pass overrides are all hardcoded. This blocks:

- Adding new material types without writing new loader code
- Artist/tool-driven material authoring
- Data-driven default value iteration

REQ-031 and REQ-032 have established the ownership contract and pass-aware material interface. The missing piece is an external asset format that describes defaults and metadata on top of the reflection-driven runtime.

## What Changes

- Add `yaml-cpp` dependency via FetchContent
- Define a YAML material asset schema (`*.mat.yaml`) supporting:
  - Shader reference (name or path)
  - Global default parameter values (by `bindingName.memberName`)
  - Global default resource references (texture paths or built-in placeholders like `white`, `black`, `normal`)
  - Per-pass parameter/resource overrides under `passes.<passName>`
  - Shader variant declarations per pass
- Implement `GenericMaterialLoader` that reads a `.mat.yaml`, compiles the shader, builds template + instance, and applies defaults — no material-specific C++ code needed
- Add built-in placeholder textures: `white` (1x1 RGBA white), `black` (1x1 RGBA black), `normal` (1x1 flat normal)
- Existing `loadBlinnPhongMaterial()` remains as-is (not replaced, but the generic loader can express the same material)

## Capabilities

### New Capabilities
- `material-asset-loader`: YAML material asset schema, generic material loader, built-in placeholder textures

### Modified Capabilities
_None — existing material system APIs are consumed as-is._

## Impact

- New dependency: `yaml-cpp` (FetchContent)
- New files: `src/infra/material_loader/generic_material_loader.{hpp,cpp}`
- New files: `src/infra/texture_loader/placeholder_textures.{hpp,cpp}`
- `src/infra/CMakeLists.txt` — add sources and yaml-cpp link
- `notes/subsystems/material-system.md` — add generic loader section
