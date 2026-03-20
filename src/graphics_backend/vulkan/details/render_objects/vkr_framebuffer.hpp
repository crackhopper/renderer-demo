#pragma once
#include <vulkan/vulkan.h>
#include <vector>
namespace LX_core::graphic_backend {
class VulkanDevice;
/**
 * @brief 帧缓冲包装（RAII，随 Swapchain 或 RenderTarget 销毁）
 */
class VulkanFrameBuffer {
public:
  VulkanFrameBuffer(VulkanDevice &device, VkRenderPass rp,
                    const std::vector<VkImageView> &attachments,
                    VkExtent2D extent);
  ~VulkanFrameBuffer();

  VkFramebuffer getHandle() const;
  VkExtent2D getExtent() const;

private:
  VkFramebuffer m_handle;
};

} // namespace LX_core::graphic_backend