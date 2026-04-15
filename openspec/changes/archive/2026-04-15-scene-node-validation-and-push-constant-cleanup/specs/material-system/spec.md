## ADDED Requirements

### Requirement: Engine-wide draw push constant ABI is model-only
The renderer SHALL use a single engine-wide draw push constant ABI consisting only of the model transform payload:

`struct alignas(16) PC_Base { Mat4f model; };`

If `PC_Draw` remains in the codebase as a compatibility alias or extension point, it MUST NOT add `enableLighting`, `enableSkinning`, or any other field that changes shader interface or pipeline shape. Shader-side push constant blocks used by the forward material path MUST match this model-only ABI exactly.

#### Scenario: Forward shader uses model-only push constant
- **WHEN** the forward material path compiles `blinnphong_0` shaders
- **THEN** the push constant block layout matches the engine-wide model-only ABI and contains no skinning or lighting feature flags

### Requirement: MaterialTemplate owns shader variants per pass
Shader variants that change shader code shape or pipeline identity SHALL belong to `MaterialTemplate` / loader output, not to `MaterialInstance`. For each configured pass, the loader MUST determine the enabled variant set, pass that set into shader compilation, and persist the same set in `RenderPassEntry::shaderSet.variants`.

`MaterialInstance` SHALL continue to own only runtime instance parameters such as UBO values, textures, and pass enable state. Instance-level parameter writes MUST NOT introduce a new variant identity dimension.

#### Scenario: Loader persists a skinning variant on the template pass
- **WHEN** a loader creates a material template for a pass that enables skinning
- **THEN** the pass's shader compilation input and stored `RenderPassEntry::shaderSet.variants` both include the skinning variant

#### Scenario: Runtime parameter writes do not create variants
- **WHEN** a `MaterialInstance` updates UBO values or textures for an existing pass
- **THEN** the template-owned variant set for that pass remains unchanged
