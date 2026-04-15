## MODIFIED Requirements

### Requirement: Leaf resources provide parameterless getRenderSignature
Every leaf resource participating in pipeline identity SHALL provide a `StringID getRenderSignature() const` method that returns a structurally-interned `StringID` computed via `GlobalStringTable::compose(...)` or `GlobalStringTable::Intern(...)`. This applies to `VertexLayoutItem`, `VertexLayout`, `RenderState`, `ShaderProgramSet`, and `RenderPassEntry`.

`PrimitiveTopology` SHALL be handled by a free function `StringID topologySignature(PrimitiveTopology)` instead of a member, since enums cannot carry methods.

#### Scenario: Same VertexLayout produces same signature
- **WHEN** two `VertexLayout` instances have identical items and stride
- **THEN** `getRenderSignature()` returns the same `StringID` id on both

#### Scenario: Different layout item order produces different signature
- **WHEN** two `VertexLayout` instances share the same items but in different declaration order
- **THEN** `getRenderSignature()` returns different `StringID` ids (compose is order sensitive)

#### Scenario: RenderState signature reflects all seven render-state fields
- **WHEN** any of `cullMode`, `depthTestEnable`, `depthWriteEnable`, `depthOp`, `blendEnable`, `srcBlend`, `dstBlend` changes
- **THEN** `RenderState::getRenderSignature()` returns a different `StringID` id

#### Scenario: ShaderProgramSet signature ignores variant order
- **WHEN** two `ShaderProgramSet` instances enable the same set of variant macros in different insertion order
- **THEN** `getRenderSignature()` returns the same id (variants MUST be sorted before compose)

#### Scenario: Topology signature distinguishes primitive types
- **WHEN** `topologySignature(PrimitiveTopology::TriangleList)` and `topologySignature(PrimitiveTopology::LineList)` are compared
- **THEN** the returned `StringID` ids are different

### Requirement: IRenderable exposes pass-aware getRenderSignature
`IRenderable` SHALL declare `virtual StringID getRenderSignature(StringID pass) const = 0`. `SceneNode`, as the primary high-level implementation, SHALL override it as `compose(TypeTag::ObjectRender, {meshSig})` where `meshSig = mesh->getRenderSignature(pass)`. Optional `Skeleton` presence MUST NOT contribute a separate field to the object render signature.

#### Scenario: Skeleton presence does not change object signature
- **WHEN** the same `SceneNode` is queried for `getRenderSignature(Pass_Forward)` before and after a `Skeleton` is attached, with the same mesh and material variants
- **THEN** the returned `StringID` id is unchanged

#### Scenario: Mesh change still changes object signature
- **WHEN** a `SceneNode` switches to a mesh with a different render signature and `getRenderSignature(Pass_Forward)` is queried again
- **THEN** the returned `StringID` id differs
