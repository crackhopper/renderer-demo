# Graphics Backend TODO

> 基于 core 层重构后的 `IRenderResource` / `IRenderable` / `RenderItem` / `Scene` 接口，
> 对 Vulkan 后端进行适配。以下按优先级和依赖关系排列。

---

## Phase 1: 基础对齐 — 修复编译 & 接口适配

### 1.1 VulkanRenderer 接口对齐
当前 `VulkanRenderer` 的接口（`uploadMesh`, `uploadTexture`, `setCamera`, `drawMesh`, `flush`）
与新的 `core::gpu::Renderer` 接口（`initialize`, `shutdown`, `initScene`, `uploadData`, `draw`）不匹配。

- [ ] 移除旧接口方法（`uploadMesh`, `uploadTexture`, `setCamera`, `drawMesh`, `flush`）
- [ ] 实现新接口：
  - `initScene(ScenePtr)` — 根据 Scene 持有的 RenderItem 创建所有 GPU 资源
  - `uploadData()` — 遍历 dirty 资源，上传/更新到 GPU
  - `draw()` — 录制命令缓冲 + 提交

### 1.2 修复旧 include 路径
以下文件引用了已不存在的 core 层头文件，需更新：

| 文件 | 旧 include | 改为 |
|------|-----------|------|
| `vkp_blinnphong.hpp` | `core/resources/vertex.hpp` | `core/resources/vertex_buffer.hpp` |
| `vks_slot.hpp` | `core/resources/schema.hpp` | `core/gpu/render_resource.hpp`（使用 `PipelineSlotId` / `ResourceType`） |
| `vkb_skeleton.hpp` | `core/resources/skeleton.hpp` | `core/scene/components/skeleton.hpp` |

### 1.3 移除旧 DrawCommand
`VulkanRenderer` 中的 `DrawCommand{MeshPtr, MaterialPtr}` 已被 core 层的 `RenderItem` 取代，删除。

---

## Phase 2: 资源创建 — 从 RenderItem 驱动 GPU 资源

核心思路：后端遍历 `RenderItem` 中的 `IRenderResourcePtr` 列表，根据 `getType()` 和 `getPipelineSlotId()`
分发到对应的 Vulkan 资源创建逻辑。

### 2.1 ResourceManager — 资源句柄映射
新建 `VulkanResourceManager`（或直接在 `VulkanRenderer` 中），维护 `IRenderResource* → VkResource` 的映射。

- [ ] 用 `getResourceHandle()`（即 `IRenderResource` 的地址）作为 key
- [ ] 支持的资源类型映射：
  - `ResourceType::VertexBuffer` → `VulkanBuffer`（VB）
  - `ResourceType::IndexBuffer` → `VulkanBuffer`（IB）
  - `ResourceType::UniformBuffer` → `VulkanUniformBuffer`
  - `ResourceType::CombinedImageSampler` → `VulkanTexture`
  - `ResourceType::PushConstant` → 无需 GPU 对象，`vkCmdPushConstants` 直接使用 `getRawData()`
  - `ResourceType::VertexShader` / `FragmentShader` → `VulkanShaderModule`（通过 `Shader::getShaderName()` 加载 SPIR-V）

### 2.2 改造 VulkanMesh
当前 `VulkanMesh::upload()` 接受旧的 `LX_core::Mesh&`。改为接受两个 `IRenderResourcePtr`：

- [ ] `createFromResource(VulkanDevice&, IRenderResourcePtr vb, IRenderResourcePtr ib)`
- [ ] 内部通过 `getRawData()` / `getByteSize()` 创建 staging buffer → device local buffer
- [ ] 保留 `indexCount` 供 `vkCmdDrawIndexed` 使用（从 `IndexBuffer::indexCount()` 获取，
      需要在 `IRenderResource` 上或通过 `dynamic_cast` 获取）

### 2.3 改造 VulkanTexture
当前 `VulkanTexture::upload()` 接受 `LX_core::Texture&`。改为接受 `CombinedTextureSampler`：

- [ ] 通过 `CombinedTextureSampler::texture()->desc()` 获取 width/height/format
- [ ] 通过 `getRawData()` / `getByteSize()` 获取像素数据
- [ ] `TextureFormat → VkFormat` 转换函数

### 2.4 Shader 加载
当前 pipeline 硬编码 shader 路径。改为通过 `Shader::getShaderName()` 查找 SPIR-V 文件：

- [ ] 建立 `shaderName → SPIR-V 文件路径` 的映射（约定或配置）
- [ ] `VulkanShaderModule` 按名字加载，缓存已加载的模块

---

## Phase 3: Descriptor Set — 基于 PipelineSlotId 绑定

### 3.1 统一 Descriptor Set Layout
当前 `DescriptorSetLayoutIndex`（Camera=0, Light=1, Material=2, Skeleton=3）需与
core 层的 `PipelineSlotId` 对齐。建立映射：

| PipelineSlotId | set | binding | VkDescriptorType |
|---------------|-----|---------|-------------------|
| CameraUBO | 0 | 0 | UNIFORM_BUFFER |
| LightUBO | 1 | 0 | UNIFORM_BUFFER |
| MaterialUBO | 2 | 0 | UNIFORM_BUFFER |
| AlbedoTexture | 2 | 1 | COMBINED_IMAGE_SAMPLER |
| NormalTexture | 2 | 2 | COMBINED_IMAGE_SAMPLER |
| BoneUBO | 3 | 0 | UNIFORM_BUFFER |

- [ ] 建立 `PipelineSlotId → (set, binding, descriptorType)` 的查表函数
- [ ] 替换 `vks_slot.hpp` 中旧的 `SlotId` / `SlotType` 为 `PipelineSlotId` / `ResourceType`

### 3.2 通用 ResourceBinding
当前 `ResourceBindingBase<Derived, Data>` 基于具体 core 类型做模板特化
（`CameraResourceBinding`, `MaterialResourceBinding`, `SkeletonResourceBinding`）。
改为统一的 `IRenderResource` 驱动方式：

- [ ] 新建 `VulkanDescriptorSetWriter`：接受 `vector<IRenderResourcePtr>` 中类型为
      `UniformBuffer` 或 `CombinedImageSampler` 的资源，按 `getPipelineSlotId()` 写入对应 binding
- [ ] 每个 descriptor set 按 set index 分组，一次 `vkUpdateDescriptorSets` 写入

### 3.3 Camera / Light UBO 绑定
Camera 和 Light 的 UBO 作为全局资源（per-scene），不通过 RenderItem 传递：

- [ ] `initScene()` 时从 `Scene::camera` / `Scene::directionalLight` 获取 UBO
- [ ] 创建对应的 `VulkanUniformBuffer`，分配 descriptor set（set 0 / set 1）
- [ ] `uploadData()` 时检查 dirty，`VulkanUniformBuffer::update()` 上传
- [ ] `vk_camera.hpp` 目前为空，实现 Camera UBO 的创建和更新

---

## Phase 4: Pipeline — 基于 VertexFormat + ShaderName 构建

### 4.1 Pipeline Cache / Registry
当前只有一个硬编码的 `VulkanPipelineBlinnPhong`。改为按 `(shaderName, vertexFormat)` 动态创建/缓存：

- [ ] `VulkanPipelineRegistry`：`(shaderName, vertexFormat) → VulkanPipelineBase*` 缓存
- [ ] 在 `initScene()` 或首次 draw 时，根据 `RenderItem::vertexShader/fragmentShader` 和
      `RenderItem::vertexFormat` 查找或创建 pipeline

### 4.2 VertexFormat → VkVertexInputState
当前 `VulkanPipelineBlinnPhong` 硬编码顶点输入。改为根据 `VertexFormat` 枚举生成：

- [ ] `VertexFormat → vector<VkVertexInputAttributeDescription>` 转换函数
- [ ] 每种 VertexFormat 对应的 stride 和 attribute 描述

### 4.3 Pipeline Layout 对齐
Pipeline layout 的 `setLayouts[]` 和 `pushConstantRanges` 需要与 Phase 3 的 descriptor set layout 一致：

- [ ] push constant range = `sizeof(ObjectPC::Param)`，stage = VERTEX
- [ ] 4 个 descriptor set layout 按 set index 排列

---

## Phase 5: 绘制循环 — RenderItem 驱动录制

### 5.1 initScene() 实现
```
initScene(ScenePtr scene):
  1. scene->buildRenderItem() 得到 RenderItem
  2. 遍历 RenderItem 的每个 IRenderResourcePtr，按 ResourceType 创建 GPU 资源（Phase 2）
  3. 创建 Camera / Light 的 GPU 资源（Phase 3.3）
  4. 查找/创建 Pipeline（Phase 4）
  5. 分配并写入 Descriptor Sets（Phase 3.2）
```

- [ ] 实现以上流程

### 5.2 uploadData() 实现
```
uploadData():
  遍历所有已注册的 IRenderResourcePtr：
    if isDirty():
      根据 ResourceType 更新对应 GPU 资源（buffer update / image re-upload）
      clearDirty()
```

- [ ] 实现 dirty 检查 + 增量上传

### 5.3 draw() 实现
```
draw():
  acquireNextImage()
  beginCommandBuffer()
  beginRenderPass()
    bindPipeline(当前 pipeline)
    bindDescriptorSet(set 0, camera)
    bindDescriptorSet(set 1, light)
    for each RenderItem:
      pushConstants(RenderItem.pushConstant)
      bindDescriptorSet(set 2, material descriptors)
      bindDescriptorSet(set 3, skeleton descriptors)  // if present
      bindVertexBuffer(RenderItem.vertexBuffer)
      bindIndexBuffer(RenderItem.indexBuffer)
      drawIndexed(indexCount)
  endRenderPass()
  endCommandBuffer()
  submit + present
```

- [ ] 实现以上录制逻辑
- [ ] 多帧飞行（MAX_FRAMES_IN_FLIGHT=2）的 per-frame 资源管理

---

## Phase 6: 清理 & 后续

### 6.1 删除废弃代码
- [ ] 删除旧的 `DrawCommand` 结构
- [ ] 删除 `m_meshMap` / `m_textureMap`（被 ResourceManager 替代）
- [ ] 删除 `details/bindings/` 目录下旧的 `vkb_camera.hpp` / `vkb_material.hpp`
      （如已被新的通用 DescriptorSetWriter 替代）
- [ ] 清理 `VulkanRenderer` 中旧的 private 方法

### 6.2 多 RenderItem 支持
当前 Scene 只有一个 mesh。后续扩展为 `vector<RenderItem>`：

- [ ] `Scene::buildRenderItems()` 返回多个 RenderItem
- [ ] draw loop 遍历所有 RenderItem
- [ ] 按 pipeline 排序减少状态切换

### 6.3 资源生命周期
- [ ] 资源销毁通知（当 `IRenderResource` 的 `shared_ptr` 析构时，清理对应 GPU 资源）
- [ ] 考虑引入 `weak_ptr` 观察或回调机制

### 6.4 Depth Buffer
- [ ] 确保 render pass 包含 depth attachment
- [ ] 创建 depth image + image view

### 6.5 Swapchain 重建
- [ ] 窗口 resize 时重建 swapchain、framebuffer、pipeline viewport
