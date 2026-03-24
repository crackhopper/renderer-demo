## Context

The Vulkan backend follows a hierarchical ownership model where `VulkanDevice` is the top-level owner that creates and owns dependent objects (descriptor manager, command buffer manager, resource manager, etc.). However, the current implementation uses raw pointers for non-owning references, which:

1. Obscures lifetime requirements (is this object owned or just referenced?)
2. Risks dangling pointers if lifetime invariants are violated
3. Violates the project's stated C++ design policy on raw pointer prohibition

## Goals / Non-Goals

**Goals:**
- Eliminate raw pointers used to hold references to other objects (not handles/identifiers)
- Make ownership semantics explicit through type system
- Apply consistent rules across all Vulkan backend classes
- Add enforceable policy to the C++ style guide

**Non-Goals:**
- Changing handle types (VkDevice, VkBuffer, etc. remain as-is - these are opaque handles, not pointers)
- Modifying the overall architecture or ownership hierarchy
- Retrofitting `shared_ptr` everywhere - most dependencies are actually non-owning references

## Decisions

### 1. Device Reference Policy

**Decision**: Replace `VulkanDevice*` with `VulkanDevice&` for all non-owning device references.

**Rationale**:
- `VulkanDevice` is the top-level owner in the Vulkan subsystem - nothing owns it (created at app startup, destroyed at shutdown)
- All dependent objects (command buffers, descriptors, resources) have shorter lifetimes than the device
- The device reference is always required (non-nullable) - using a reference enforces this invariant
- Using a reference rather than `shared_ptr` because no cyclic dependency exists

**Affected Classes**:
- `VulkanCommandBuffer` (`m_device`)
- `VulkanCommandBufferManager` (`m_device`)
- `VulkanDescriptorManager` (`m_device`)
- `VulkanPipelineBase` (`m_device`)
- `VulkanRenderPass` (`m_device`)
- `VulkanFrameBuffer` (`m_device`)
- `VulkanSwapchain` (`m_device`)

### 2. Injected Dependency Policy

**Decision**: All dependencies MUST be injected via constructor and stored as references (`T&`). Setter methods for dependencies are prohibited. This ensures all dependencies are initialized before use and eliminates nullable states.

**Implementation Requirements**:
- Constructor MUST accept all required dependencies as reference parameters
- Dependencies MUST be initialized in the constructor initializer list
- No setter methods for dependency injection
- No raw pointers for storing dependencies

**Affected Classes**:
- `VulkanCommandBuffer` - change `setResourceManager(VulkanResourceManager*)` to constructor injection `VulkanCommandBuffer(VulkanDevice&, VulkanResourceManager&)`
- `VulkanResourceManager` - change `setCommandBufferManager(VulkanCommandBufferManager*)` to constructor injection `VulkanResourceManager(VulkanDevice&, VulkanCommandBufferManager&)`
- `DescriptorSet` - already uses `VulkanDescriptorManager&` - correct, no change needed

**Rationale**:
- Constructor injection makes dependencies explicit and guaranteed at construction time
- No nullable states means no null pointer checks needed
- Easier to reason about object invariants
- setter patterns allow objects to exist in incomplete/invalid states temporarily

### 3. VulkanResourceManager Ownership

**Decision**: `VulkanResourceManager` should own its render pass, pipeline, and command buffer manager via `std::unique_ptr`.

**Rationale**:
- Currently `m_renderPass`, `m_pipeline`, `m_cmdBufferMgr` are raw pointers with unclear ownership
- These objects are created by the resource manager and should be destroyed when the resource manager is destroyed
- This follows the "unique ownership" pattern - if an object is created and owned by another class, use `unique_ptr`

**Changes**:
- `m_renderPass`: `VulkanRenderPass*` → `std::unique_ptr<VulkanRenderPass>`
- `m_pipeline`: `VulkanPipelineBase*` → `std::unique_ptr<VulkanPipelineBase>`
- `m_cmdBufferMgr`: `VulkanCommandBufferManager*` → `std::unique_ptr<VulkanCommandBufferManager>`
- Remove setter methods, create objects directly in `VulkanResourceManager`

### 4. Exceptions for Handle Types

**Decision**: Raw pointers used as handles or identifiers (e.g., lookup keys in maps) are acceptable and do not need refactoring.

**Rationale**:
- `void*` used as lookup keys in `m_gpuResources` map - these are identifiers, not references to objects
- The map stores `std::shared_ptr<VulkanAnyResource>` for the actual object lifetime
- These pointers are never dereferenced to access object state

## Ownership Model After Refactoring

```
VulkanDevice (top-level owner)
├── unique_ptr<VulkanDescriptorManager> m_descriptorManager
├── unique_ptr<VulkanCommandBufferManager> m_cmdBufferManager (moved from ResourceManager)
└── unique_ptr<VulkanResourceManager> m_resourceManager
    ├── unique_ptr<VulkanRenderPass> m_renderPass
    ├── unique_ptr<VulkanPipelineBase> m_pipeline
    └── unique_ptr<VulkanCommandBufferManager> m_cmdBufferMgr

VulkanSwapchain (independent, receives device reference)
├── vector<unique_ptr<VulkanFrameBuffer>> m_framebuffers
└── VulkanDevice& m_device (reference, not owned)

Other Vulkan objects (created by factories, receive device reference)
├── VulkanCommandBufferManager& m_device
├── VulkanDescriptorManager& m_device
├── VulkanRenderPass& m_device
├── VulkanFrameBuffer& m_device
├── VulkanCommandBuffer& m_device
└── VulkanPipelineBase& m_device
```

## Risks / Trade-offs

- **[Risk] Breaking API changes**: Changing constructor signatures (pointer → reference) will require updates to all call sites. [Mitigation] Use consistent factory patterns and update all tests.

- **[Risk] Move-only objects**: `std::unique_ptr` cannot be copied, which may affect object handling in containers or during construction. [Mitigation] Design constructors to take `std::unique_ptr` by move, not by copy.

- **[Trade-off] Increased header includes**: References must be initialized in constructor initializer lists, which may require including more headers. This is acceptable for clearer ownership semantics.

## Migration Plan

1. First, update `VulkanResourceManager` to own its dependent objects (`m_renderPass`, `m_pipeline`, `m_cmdBufferMgr`)
2. Update all other classes to use `VulkanDevice&` instead of `VulkanDevice*`
3. Update test files to use new constructor signatures
4. Update the C++ style guide with the new policy
5. Verify compilation and run tests
