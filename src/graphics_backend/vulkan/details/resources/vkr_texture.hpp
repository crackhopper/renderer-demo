#pragma once
#include <memory>
#include <vulkan/vulkan.h>

namespace LX_core::graphic_backend {

class VulkanCommandBuffer; 
class VulkanBuffer;
class VulkanDevice;

 
class VulkanTexture;
using VulkanTexturePtr = std::unique_ptr<VulkanTexture>;

class VulkanTexture {
  struct Token {};

public:
  VulkanTexture(Token, const VulkanDevice &_device, uint32_t width,
                uint32_t height, VkFormat format, VkImageUsageFlags usage,
                VkFilter filter);
  ~VulkanTexture();

  static VulkanTexturePtr create(const VulkanDevice &_device, uint32_t width,
                                 uint32_t height, VkFormat format,
                                 VkImageUsageFlags usage,
                                 VkFilter filter = VK_FILTER_LINEAR) {
    return std::make_unique<VulkanTexture>(Token{}, _device, width, height,
                                           format, usage, filter);
  }

  // 用于 Descriptor Set 绑定的信息
  VkDescriptorImageInfo getDescriptorInfo() const {
    return {sampler, imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  }

  VkImage getHandle() const { return image; }

  void transitionLayout(VulkanCommandBuffer &cmd, VkImageLayout oldLayout,
                        VkImageLayout newLayout);
  void copyFromBuffer(VulkanCommandBuffer &cmd, class VulkanBuffer &buffer);  

private:
  VkDevice device = VK_NULL_HANDLE;
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkImageView imageView = VK_NULL_HANDLE;
  VkSampler sampler = VK_NULL_HANDLE;


};

} // namespace LX_core::graphic_backend