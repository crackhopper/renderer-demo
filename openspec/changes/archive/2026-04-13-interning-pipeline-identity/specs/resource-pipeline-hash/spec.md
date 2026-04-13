## REMOVED Requirements

### Requirement: Mesh exposes getPipelineHash

**Reason**: Pipeline identity is now driven by `Mesh::getRenderSignature(pass)` (a structurally composed `StringID`) per the `render-signature` capability, not by a `size_t` hash. `Mesh::getLayoutHash()` is retained as a generic hash-map helper but no longer participates in pipeline identity.
**Migration**: Replace callers of `mesh.getPipelineHash()` with `mesh.getRenderSignature(pass)`. For internal `std::unordered_map<..., size_t>` uses, continue to call `getLayoutHash()`.

### Requirement: RenderState exposes getPipelineHash

**Reason**: `RenderState::getRenderSignature()` now supplies pipeline identity as a structured `StringID`. `RenderState::getHash()` remains for hash-map uses.
**Migration**: Replace `state.getPipelineHash()` with `state.getRenderSignature()` at pipeline-identity sites; keep `getHash()` for hash-map keys.

### Requirement: ShaderProgramSet exposes getPipelineHash

**Reason**: `ShaderProgramSet::getRenderSignature()` supplies pipeline identity (variants sorted before compose). `ShaderProgramSet::getHash()` remains for internal lookups.
**Migration**: Replace `shaderSet.getPipelineHash()` with `shaderSet.getRenderSignature()` at pipeline-identity sites.

### Requirement: Skeleton exposes getPipelineHash for skinning

**Reason**: Replaced by `Skeleton::getRenderSignature()` returning `Intern("Skn1")`. The `kSkeletonPipelineHashTag` constant is also removed.
**Migration**: Replace `skeleton->getPipelineHash()` with `skeleton->getRenderSignature()`; at callers that only need "does skinning apply," use `StringID{}` (id 0) sentinel for the no-skeleton case as `RenderableSubMesh::getRenderSignature` does.

### Requirement: Future PipelineKey assembly uses getPipelineHash

**Reason**: Superseded by the new `pipeline-key` requirement "PipelineKey build composes object and material signatures" which mandates `compose(TypeTag::PipelineKey, {objectSig, materialSig})` instead of numeric hash combination.
**Migration**: `PipelineKey::build` is now `build(StringID objectSig, StringID materialSig)`. See `render-signature` capability for how callers resolve those signatures via `IRenderable::getRenderSignature(pass)` and `IMaterial::getRenderSignature(pass)`.
