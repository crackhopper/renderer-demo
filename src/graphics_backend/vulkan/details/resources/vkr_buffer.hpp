#pragma once

#include <memory>
#include <vulkan/vulkan.h>

namespace LX_core::graphic_backend {

class VulkanDevice;
class VulkanCommandBuffer;


// 这个资源是一个 Vulkan 缓冲区，用于存储：
// - 顶点数据
// - 索引数据
// - Uniform 数据
class VulkanBuffer;
using VulkanBufferPtr = std::unique_ptr<VulkanBuffer>;
class VulkanBuffer {
  struct Token {};

public:
  VulkanBuffer(Token token, VulkanDevice &_device, VkDeviceSize _size,
               VkBufferUsageFlags _usage, VkMemoryPropertyFlags properties);
  ~VulkanBuffer();

  static VulkanBufferPtr create(const VulkanDevice &_device, VkDeviceSize size,
                                VkBufferUsageFlags usage,
                                VkMemoryPropertyFlags properties) {
    return std::make_unique<VulkanBuffer>(Token{}, _device, size, usage,
                                          properties);
  }

  void *map();
  void unmap();

  void uploadData(const void *data, VkDeviceSize dataSize);

  void copyTo(VulkanCommandBuffer &cmdBuffer, VulkanBuffer &dst);

  VkBuffer getHandle() const { return buffer; }

private:
  VkDevice device = VK_NULL_HANDLE;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  VkBufferUsageFlags usage;
};
} // namespace LX_core::graphic_backend