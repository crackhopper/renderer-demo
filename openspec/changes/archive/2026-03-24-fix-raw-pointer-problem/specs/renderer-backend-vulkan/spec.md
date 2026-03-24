# Delta: renderer-backend-vulkan Specification

## Section: Object Ownership Model

### Additional Requirement: Vulkan objects shall use explicit ownership semantics

All Vulkan backend classes SHALL follow these ownership rules:

1. **`VulkanDevice` as Top-Level Owner**: `VulkanDevice` owns all Vulkan subsystem objects via `std::unique_ptr`. No other class owns the device.

2. **Non-Owning Device References**: All classes that reference `VulkanDevice` (but do not own it) SHALL use `VulkanDevice&` (reference) instead of `VulkanDevice*` (pointer).

3. **Non-Owning References for Short-Lived Objects**: Classes with shorter lifetimes than the device (e.g., `VulkanCommandBuffer`, `VulkanRenderPass`, `VulkanFrameBuffer`) receive device references in constructors.

4. **`VulkanResourceManager` Ownership**: `VulkanResourceManager` owns its render pass, pipeline, and command buffer manager via `std::unique_ptr`. These are not shared objects.

5. **No Raw Pointers for Object References**: Raw pointers (`T*`) SHALL NOT be used to hold references to other classes except:
   - `void*` as lookup keys in resource maps
   - Pointers to opaque Vulkan handles (VkDevice, VkBuffer, etc.)

#### Scenario: Command buffer references device
- **WHEN** `VulkanCommandBuffer` is constructed
- **THEN** It SHALL receive `VulkanDevice&` as a reference (not pointer)
- **AND** The reference SHALL be used for all device operations
- **AND** The command buffer does not own the device

#### Scenario: Resource manager owns pipeline
- **WHEN** `VulkanResourceManager` is constructed
- **THEN** It SHALL create and own its `VulkanPipelineBase` via `std::unique_ptr<VulkanPipelineBase>`
- **AND** The pipeline SHALL be destroyed when the resource manager is destroyed
- **AND** No other class holds a owning pointer to this pipeline

### Additional Requirement: Reference types shall be used for non-nullable dependencies

When a class requires a reference to another object that is guaranteed to outlive it:

1. **Constructor Injection**: Dependencies SHALL be passed via constructor and stored as references (`T&`).
2. **Setter Injection**: Dependencies that cannot be provided at construction (optional or set from factory) MAY be set via setter methods using raw pointers with documentation.
3. **No Raw Pointers for Required Dependencies**: Do not use `T*` when `T&` correctly expresses the non-nullable invariant.

#### Scenario: DescriptorSet requires manager reference
- **WHEN** `DescriptorSet` is constructed
- **THEN** It SHALL receive `VulkanDescriptorManager&` as a reference
- **AND** The reference SHALL be stored as `VulkanDescriptorManager& m_manager`
