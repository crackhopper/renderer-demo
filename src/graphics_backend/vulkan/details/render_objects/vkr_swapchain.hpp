#pragma once
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

namespace LX_core::graphic_backend {

class VulkanDevice;
class VulkanRenderPass;
class VulkanFrameBuffer;

/**
 * @brief 交换链封装类
 * 管理显示图像队列、渲染目标同步以及 Framebuffer 的生命周期
 */
class VulkanSwapchain {
  struct Token {};

public:
  VulkanSwapchain(Token, VulkanDevice &device, VkSurfaceKHR surface,
                  VkExtent2D extent, uint32_t graphicsIdx, uint32_t presentIdx, uint32_t maxFramesInFlight = 3);
  ~VulkanSwapchain();

  static std::unique_ptr<VulkanSwapchain>
  create(VulkanDevice &device, VkSurfaceKHR surface, VkExtent2D extent, uint32_t graphicsIdx, uint32_t presentIdx, uint32_t maxFramesInFlight = 3) {
    return std::make_unique<VulkanSwapchain>(Token{}, device, surface, extent, graphicsIdx, presentIdx, maxFramesInFlight);
  }

  // --- 核心生命周期控制 ---

  /**
   * @brief 当窗口大小改变或 Swapchain 失效时调用
   */
  void rebuild(VkExtent2D newExtent, VulkanRenderPass &renderPass);

  /**
   * @brief 获取当前帧可用的信号量（用于同步）
   */
  VkSemaphore getImageAvailableSemaphore(uint32_t currentFrameIndex) const;
  VkSemaphore getRenderFinishedSemaphore(uint32_t currentFrameIndex) const;
  VkFence getInFlightFence(uint32_t currentFrameIndex) const;

  /**
   * @brief 获取下一帧图像索引
   * 内部使用第 currentFrameIndex 组 ImageAvailableSemaphore
   */
  VkResult acquireNextImage(uint32_t currentFrameIndex, uint32_t &imageIndex);

  /**
   * @brief 提交显示
   * 内部使用第 currentFrameIndex 组 RenderFinishedSemaphore
   */
  VkResult present(uint32_t currentFrameIndex, uint32_t imageIndex);

  // --- 资源访问 ---

  VkSwapchainKHR getHandle() const;
  VkExtent2D getExtent() const;
  VulkanFrameBuffer &getFramebuffer(uint32_t index);
  uint32_t getImageCount() const;

  void initialize(VulkanRenderPass &renderPass){
    createInternal(m_extent);
    createImageViews();
    createDepthResources();
    createSyncObjects();
    setupFramebuffers(renderPass);
  }


private:
  VulkanDevice &m_device;
  uint32_t m_maxFramesInFlight;
  VkSurfaceKHR m_surface;

  VkSwapchainKHR m_handle = VK_NULL_HANDLE;
  VkFormat m_imageFormat;
  VkExtent2D m_extent;

  // 原始图像资源
  std::vector<VkImage> m_images;
  std::vector<VkImageView> m_imageViews;

  // 深度资源（通常 Swapchain 每个 Image 共享或独立拥有一个深度缓冲）
  VkImage m_depthImage = VK_NULL_HANDLE;
  VkDeviceMemory m_depthImageMemory = VK_NULL_HANDLE;
  VkImageView m_depthImageView = VK_NULL_HANDLE;

  // 绑定的 Framebuffers：生命周期随 Swapchain 重建而重置
  std::vector<std::unique_ptr<class VulkanFrameBuffer>> m_framebuffers;

  // --- 同步原语组 ---
  // 每组包含：一个让 CPU 等待 GPU 的 Fence，两个用于 Image 轮转的 Semaphore
  // 数组大小均为 m_maxFramesInFlight
  std::vector<VkSemaphore> m_imageAvailableSemaphores;
  std::vector<VkSemaphore> m_renderFinishedSemaphores;
  std::vector<VkFence> m_inFlightFences;




  // 辅助初始化函数
  void cleanup();
  void createInternal(VkExtent2D extent);
  void createImageViews();
  void createDepthResources();
  void createSyncObjects();
  void setupFramebuffers(VulkanRenderPass &renderPass);
};

using VulkanSwapchainPtr = std::unique_ptr<VulkanSwapchain>;

} // namespace LX_core::graphic_backend