## MODIFIED Requirements

### Requirement: Skeleton exposes skinning presence for pipeline identity
`Skeleton` SHALL NOT contribute a render-signature leaf or any other direct pipeline-identity token. The previous `StringID getRenderSignature() const`, `size_t getPipelineHash() const`, and `kSkeletonPipelineHashTag` constant SHALL NOT exist. `Skeleton` remains a runtime resource provider through its Bones UBO and a legality dependency for passes whose material variants require skinning.

#### Scenario: Skeleton no longer exposes identity helpers
- **WHEN** a caller attempts to invoke `skeleton->getRenderSignature()` or `skeleton->getPipelineHash()`
- **THEN** the call SHALL fail to compile because those identity helpers no longer exist

#### Scenario: Skeleton still provides runtime UBO data
- **WHEN** a skinned pass is structurally valid and needs Bones data at draw time
- **THEN** the render path obtains that data from the skeleton's UBO resource rather than from any pipeline-identity API
