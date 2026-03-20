#pragma once
#include <map>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

namespace LX_core::graphic_backend {

class VulkanDevice;

// 每帧的上下文，封装 Pool 和已经分配出的 Buffer
struct CommandFrameContext {
  VkCommandPool pool = VK_NULL_HANDLE;
  std::vector<VkCommandBuffer> activeBuffers; // 当前帧正在录制的
  uint32_t nextAvailableBuffer = 0; // 用于复用 Buffer，减少 allocate 次数
};

class VulkanCommandBufferManager {
  struct Token {};

public:
  VulkanCommandBufferManager(Token, VulkanDevice &device,
                             uint32_t maxFramesInFlight,
                             uint32_t queueFamilyIndex);
  ~VulkanCommandBufferManager();

  static std::unique_ptr<VulkanCommandBufferManager>
  create(VulkanDevice &device, uint32_t maxFramesInFlight,
         uint32_t queueFamilyIndex) {
    return std::make_unique<VulkanCommandBufferManager>(
        Token{}, device, maxFramesInFlight, queueFamilyIndex);
  }

  // --- 帧生命周期控制 ---

  // 每帧渲染开始前调用，重置当前帧的 Pool
  void beginFrame(uint32_t currentFrameIndex);

  // 获取一个可用的 CommandBuffer
  // 内部逻辑：如果 activeBuffers 还有没用的就直接给，没有就申请新的
  VkCommandBuffer
  allocateBuffer(VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

  // --- 工具函数 ---

  // 立即执行的命令（用于数据上传等一次性任务）
  // 这种通常使用一个独立的、带有 VK_COMMAND_POOL_CREATE_TRANSIENT_BIT 的 Pool
  VkCommandBuffer beginSingleTimeCommands();
  void endSingleTimeCommands(VkCommandBuffer commandBuffer, VkQueue queue);

private:
  VulkanDevice &m_device;
  uint32_t m_currentFrameIndex = 0;
  uint32_t m_maxFramesInFlight = 0;
  uint32_t m_queueFamilyIndex = 0;

  std::vector<CommandFrameContext> m_frameContexts;

  // 用于一次性命令的独立 Pool
  VkCommandPool m_transientPool = VK_NULL_HANDLE;

  void createPool(VkCommandPool &pool, VkCommandPoolCreateFlags flags);
};

} // namespace LX_core::graphic_backend