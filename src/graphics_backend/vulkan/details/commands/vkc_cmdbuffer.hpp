#pragma once
#include "core/gpu/render_resource.hpp"
#include "core/scene/scene.hpp"
#include <memory>
#include <vulkan/vulkan.h>

namespace LX_core::graphic_backend {

class VulkanCommandBuffer;
using VulkanCommandBufferPtr = std::unique_ptr<VulkanCommandBuffer>;

struct RenderItem;         // 渲染数据的封装，内置vertex buffer, index buffer,
                           // descriptor sets, push constants 等信息。
class VulkanPipeline;      // 对pipeline的封装。
class VulkanRenderContext; // 渲染上下文，内置render pass, framebuffer, extent, offset, clearValues 等信息。

/**
 * @brief 这是一个轻量级包装类。
 * 它不负责 VkCommandBuffer 句柄的生命周期（由 CommandBufferManager/Pool
 * 负责）。 它只负责提供符合人体工程学的錄製接口。
 */
class VulkanCommandBuffer {
public:
  VulkanCommandBuffer(VkCommandBuffer handle) : _handle(handle) {}
  ~VulkanCommandBuffer() = default;

  VkCommandBuffer getHandle() const { return _handle; }

  // --- Render Pass 相关 (建议传入具体的 RenderPass 和 Framebuffer) ---
  void beginRenderPass(VulkanRenderContext &renderContext);
  void endRenderPass() { vkCmdEndRenderPass(_handle); }

  // --- 动态状态 ---
  void setViewport(uint32_t width, uint32_t height);
  void setScissor(uint32_t width, uint32_t height);

  // --- 核心绘制逻辑：消费 RenderItem ---

  // 1. 绑定 Pipeline (Shader/State)
  void bindPipeline(VulkanPipeline &pipeline);

  // 2. 绑定资源 (Descriptor Sets & Push Constants)
  void bindResources(VulkanPipeline &pipeline, const RenderItem &item);

  // 3. 执行绘制 (Vertex/Index Buffers & Draw Call)
  void drawItem(const RenderItem &item);

  // --- 资源传输/同步指令 (一次性指令常用) ---

  /**
   * @brief 复制 Buffer 区域
   */
  void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size,
                  VkDeviceSize srcOffset = 0, VkDeviceSize dstOffset = 0) {
    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = srcOffset;
    copyRegion.dstOffset = dstOffset;
    copyRegion.size = size;
    vkCmdCopyBuffer(_handle, src, dst, 1, &copyRegion);
  }

  /**
   * @brief 复制 Buffer 到 Image (上传纹理核心)
   */
  void copyBufferToImage(VkBuffer src, VkImage dst, uint32_t width,
                         uint32_t height) {
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};

    vkCmdCopyBufferToImage(_handle, src, dst,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  }

  /**
   * @brief 图像布局转换 (非常重要，用于将 Image 从 Undefined 转为
   * ShaderReadOnly)
   */
  void pipelineBarrier(VkPipelineStageFlags srcStage,
                       VkPipelineStageFlags dstStage,
                       VkImageMemoryBarrier barrier) {
    vkCmdPipelineBarrier(_handle, srcStage, dstStage, 0, 0, nullptr, 0, nullptr,
                         1, &barrier);
  }

private:
  VkCommandBuffer _handle;
};

} // namespace LX_core::graphic_backend