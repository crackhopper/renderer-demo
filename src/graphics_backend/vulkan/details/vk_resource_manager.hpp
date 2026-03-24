#pragma once

#include "core/gpu/render_resource.hpp"
#include "pipelines/vkp_pipeline.hpp"
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include <vulkan/vulkan.h>

namespace LX_core::graphic_backend {

class VulkanDevice;
class VulkanCommandBufferManager;
class VulkanRenderPass;
class VulkanBuffer;
class VulkanTexture;
class VulkanShader;

using VulkanBufferPtr = std::unique_ptr<VulkanBuffer>;
using VulkanTexturePtr = std::unique_ptr<VulkanTexture>;
using VulkanShaderPtr = std::unique_ptr<VulkanShader>;

using VulkanAnyResource =
    std::variant<VulkanBufferPtr, VulkanTexturePtr, VulkanShaderPtr>;

class VulkanResourceManager;
using VulkanResourceManagerPtr = std::unique_ptr<VulkanResourceManager>;
class VulkanResourceManager {
  struct Token {};

public:
  explicit VulkanResourceManager(Token token, VulkanDevice &device);
  ~VulkanResourceManager();

  static VulkanResourceManagerPtr create(VulkanDevice &device) {
    auto p = std::make_unique<VulkanResourceManager>(Token{}, device);
    return p;
  }

  // 禁止拷贝
  VulkanResourceManager(const VulkanResourceManager &) = delete;
  VulkanResourceManager &operator=(const VulkanResourceManager &) = delete;

  void syncResource(VulkanCommandBufferManager &cmdBufferManager,
                    const IRenderResourcePtr &cpuRes);
  void collectGarbage();

  void initializeRenderPassAndPipeline(VkSurfaceFormatKHR surfaceFormat,
                                       VkFormat depthFormat);

  // 快捷访问接口 - lookup functions return optional references since resource
  // may not exist
  std::optional<std::reference_wrapper<VulkanBuffer>> getBuffer(void *handle);
  std::optional<std::reference_wrapper<VulkanTexture>> getTexture(void *handle);
  std::optional<std::reference_wrapper<VulkanShader>> getShader(void *handle);
  VulkanRenderPass &getRenderPass();
  VulkanPipelineBase &getRenderPipeline();

private:
  // 内部创建与更新逻辑
  std::shared_ptr<VulkanAnyResource>
  createGpuResource(const IRenderResourcePtr &cpuRes);
  void updateGpuResource(std::shared_ptr<VulkanAnyResource> &gpuRes,
                         const IRenderResourcePtr &cpuRes,
                         VulkanCommandBufferManager &cmdBufferManager);

  VulkanDevice &m_device;
  std::unordered_map<void *, std::shared_ptr<VulkanAnyResource>> m_gpuResources;
  std::unordered_set<void *> m_activeHandles;

  // 管理若干个render pass和pipeline
  // TODO: 暂时仅支持了1个
  std::unique_ptr<VulkanRenderPass> m_renderPass;
  std::unique_ptr<VulkanPipelineBase> m_pipeline;
};

} // namespace LX_core::graphic_backend