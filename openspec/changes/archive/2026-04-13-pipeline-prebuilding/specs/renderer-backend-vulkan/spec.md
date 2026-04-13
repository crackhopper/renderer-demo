## MODIFIED Requirements

### Requirement: VulkanPipeline shall create graphics pipelines with shader stages

The VulkanPipeline SHALL consume a `LX_core::PipelineBuildInfo` (from the `pipeline-build-info` capability) as its single construction input, and SHALL support:

- Creating `VkShaderModule`s from `buildInfo.stages` (bytecode only; no filesystem loads at this layer)
- Building `VkDescriptorSetLayout`s from `buildInfo.bindings` (reflected `ShaderResourceBinding` list) — one layout per distinct `set` number, each layout populated from the `(binding, type, stageFlags)` of every `ShaderResourceBinding` whose `set` matches
- Creating a pipeline layout with the derived descriptor set layouts and the `buildInfo.pushConstant` range
- Defining vertex input state from `buildInfo.vertexLayout`
- Configuring rasterization, input assembly, viewport, depth stencil, color blending from `buildInfo.renderState` and `buildInfo.topology`
- Creating the final `VkPipeline` via `vkCreateGraphicsPipelines`

The pipeline class SHALL NOT accept any legacy `PipelineSlotDetails` vector and SHALL NOT key descriptor layout construction on hardcoded slot enums.

#### Scenario: Dynamic graphics pipeline creation from PipelineBuildInfo

- **WHEN** A caller invokes pipeline creation with a `PipelineBuildInfo` whose `bindings` contains a UBO at set 0 binding 0, a sampler at set 2 binding 1, and whose `vertexLayout` includes position+normal+uv
- **THEN** The constructed `VulkanPipeline` exposes exactly the descriptor set layouts and vertex input attributes described by the build info, with no reference to any shader-name lookup table

### Requirement: VulkanResourceManager shall map IRenderResource to Vulkan objects

The VulkanResourceManager SHALL support:
- Creating VulkanBuffer from IRenderResource with type VertexBuffer or IndexBuffer
- Creating VulkanTexture from IRenderResource with type CombinedImageSampler
- Maintaining map of IRenderResource* to created Vulkan objects
- Initializing render pass with correct formats
- Delegating pipeline caching to a standalone `LX_core::backend::PipelineCache` instance (see the `pipeline-cache` capability). The resource manager SHALL NOT store the pipeline map directly; the legacy `getOrCreateRenderPipeline(item)` helper, if retained, SHALL be a thin forward to `PipelineCache::getOrCreate(PipelineBuildInfo::fromRenderingItem(item), renderPass)`

#### Scenario: Resource mapping for vertex buffer
- **WHEN** initScene creates GPU resource for vertex buffer IRenderResource
- **THEN** VulkanResourceManager SHALL store mapping and return valid VulkanBuffer

#### Scenario: Pipeline lookup delegates to PipelineCache

- **WHEN** draw logic requests a pipeline for a `PipelineKey` present in the cache
- **THEN** the request resolves via `PipelineCache::find` and no code path references `VulkanResourceManager::m_pipelines` (which SHALL not exist after this change)

### Requirement: VulkanCommandBuffer shall record rendering commands

The VulkanCommandBuffer SHALL support:
- Beginning render pass with specified framebuffer and render pass
- Setting viewport to framebuffer dimensions
- Setting scissor to framebuffer dimensions
- Binding graphics pipeline
- Binding vertex and index buffers
- Binding descriptor sets by matching each descriptor binding from the pipeline's reflected `ShaderResourceBinding` list against the `RenderingItem::descriptorResources` via `IRenderResource::getBindingName()` (introduced by this change). The matching SHALL NOT use `PipelineSlotId`; that enum SHALL not exist after this change.
- Drawing indexed primitives with push constants

#### Scenario: Recording a draw binds descriptors by name
- **WHEN** Recording commands for a `RenderingItem` whose pipeline exposes a `"CameraUBO"` binding at set 1 binding 0 and whose `descriptorResources` contains a resource whose `getBindingName()` returns `StringID("CameraUBO")`
- **THEN** The command buffer updates the descriptor set at `(set=1, binding=0)` to reference that resource's GPU buffer, with no lookup table from a hardcoded enum

