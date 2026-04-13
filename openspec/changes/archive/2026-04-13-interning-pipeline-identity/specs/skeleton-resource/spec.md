## MODIFIED Requirements

### Requirement: Skeleton exposes skinning presence for pipeline identity

`Skeleton` SHALL provide `StringID getRenderSignature() const` returning a stable leaf id, specifically `GlobalStringTable::get().Intern("Skn1")`. The existence of a `Skeleton` instance implies skinning is active; callers that need a "no skinning" marker SHALL use the default `StringID{}` (id 0) sentinel. The previous `size_t getPipelineHash() const` method and the `kSkeletonPipelineHashTag` constant SHALL be removed.

#### Scenario: Signature is stable across instances

- **WHEN** two `Skeleton` instances are created (irrespective of their internal bone data)
- **THEN** `getRenderSignature()` SHALL return the same `StringID` id on both

#### Scenario: getPipelineHash no longer compiles

- **WHEN** a caller attempts to invoke `skeleton->getPipelineHash()`
- **THEN** the call SHALL fail to compile (method removed)
