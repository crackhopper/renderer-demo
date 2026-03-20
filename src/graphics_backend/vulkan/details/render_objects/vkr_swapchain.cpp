#include "vkr_swapchain.hpp"
#include "../vk_device.hpp"
#include "vkr_framebuffer.hpp"
#include "vkr_renderpass.hpp"
#include <algorithm>
#include <stdexcept>

namespace LX_core::graphic_backend {

VulkanSwapchain::VulkanSwapchain(Token, VulkanDevice &device,
                                 VkSurfaceKHR surface, VkExtent2D extent,
                                 uint32_t graphicsIdx, uint32_t presentIdx,
                                 uint32_t maxFramesInFlight)
    : m_device(device), m_surface(surface), m_extent(extent),
      m_maxFramesInFlight(maxFramesInFlight) {

  // 初始化时不需要立即调用 createInternal，交给 initialize 或 rebuild
}

VulkanSwapchain::~VulkanSwapchain() { cleanup(); }

void VulkanSwapchain::cleanup() {
  VkDevice logicalDevice = m_device.getHandle();

  // 1. 销毁 Framebuffers
  m_framebuffers.clear();

  // 2. 销毁同步对象
  for (size_t i = 0; i < m_maxFramesInFlight; i++) {
    vkDestroySemaphore(logicalDevice, m_imageAvailableSemaphores[i], nullptr);
    vkDestroySemaphore(logicalDevice, m_renderFinishedSemaphores[i], nullptr);
    vkDestroyFence(logicalDevice, m_inFlightFences[i], nullptr);
  }

  // 3. 销毁深度资源
  vkDestroyImageView(logicalDevice, m_depthImageView, nullptr);
  vkDestroyImage(logicalDevice, m_depthImage, nullptr);
  vkFreeMemory(logicalDevice, m_depthImageMemory, nullptr);

  // 4. 销毁 Swapchain 图像视图
  for (auto imageView : m_imageViews) {
    vkDestroyImageView(logicalDevice, imageView, nullptr);
  }

  // 5. 销毁 Swapchain 本身
  if (m_handle != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(logicalDevice, m_handle, nullptr);
  }
}

void VulkanSwapchain::createInternal(VkExtent2D extent) {
  VkPhysicalDevice physDevice = m_device.getPhysicalDevice();

  // 查询能力
  VkSurfaceCapabilitiesKHR capabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physDevice, m_surface,
                                            &capabilities);

  // 确定图像数量 (通常是 min + 1)
  uint32_t imageCount = capabilities.minImageCount + 1;
  if (capabilities.maxImageCount > 0 &&
      imageCount > capabilities.maxImageCount) {
    imageCount = capabilities.maxImageCount;
  }

  VkSwapchainCreateInfoKHR createInfo{
      VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  createInfo.surface = m_surface;
  createInfo.minImageCount = imageCount;
  createInfo.imageFormat =
      VK_FORMAT_B8G8R8A8_SRGB; // 简化处理，实际应从 findBestSurfaceFormat 传入
  createInfo.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  createInfo.imageExtent = extent;
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  // --- 处理队列族共享模式 ---
  uint32_t graphicsIdx = m_device.getGraphicsQueueFamilyIndex();
  uint32_t presentIdx = m_device.getPresentQueueFamilyIndex();
  uint32_t queueFamilyIndices[] = {graphicsIdx, presentIdx};

  if (graphicsIdx != presentIdx) {
    createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    createInfo.queueFamilyIndexCount = 2;
    createInfo.pQueueFamilyIndices = queueFamilyIndices;
  } else {
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }

  createInfo.preTransform = capabilities.currentTransform;
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR; // V-Sync 开启
  createInfo.clipped = VK_TRUE;
  createInfo.oldSwapchain = VK_NULL_HANDLE;

  if (vkCreateSwapchainKHR(m_device.getHandle(), &createInfo, nullptr,
                           &m_handle) != VK_SUCCESS) {
    throw std::runtime_error("failed to create swap chain!");
  }

  // 获取图像句柄
  vkGetSwapchainImagesKHR(m_device.getHandle(), m_handle, &imageCount,
                          nullptr);
  m_images.resize(imageCount);
  vkGetSwapchainImagesKHR(m_device.getHandle(), m_handle, &imageCount,
                          m_images.data());

  m_imageFormat = createInfo.imageFormat;
  m_extent = extent;
}

void VulkanSwapchain::createImageViews() {
  m_imageViews.resize(m_images.size());
  for (size_t i = 0; i < m_images.size(); i++) {
    VkImageViewCreateInfo createInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    createInfo.image = m_images[i];
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    createInfo.format = m_imageFormat;
    createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    createInfo.subresourceRange.baseMipLevel = 0;
    createInfo.subresourceRange.levelCount = 1;
    createInfo.subresourceRange.baseArrayLayer = 0;
    createInfo.subresourceRange.layerCount = 1;

    vkCreateImageView(m_device.getHandle(), &createInfo, nullptr,
                      &m_imageViews[i]);
  }
}

void VulkanSwapchain::createSyncObjects() {
  m_imageAvailableSemaphores.resize(m_maxFramesInFlight);
  m_renderFinishedSemaphores.resize(m_maxFramesInFlight);
  m_inFlightFences.resize(m_maxFramesInFlight);

  VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fenceInfo.flags =
      VK_FENCE_CREATE_SIGNALED_BIT; // 初始为已发出信号，防止第一帧卡死

  for (size_t i = 0; i < m_maxFramesInFlight; i++) {
    vkCreateSemaphore(m_device.getHandle(), &semaphoreInfo, nullptr,
                      &m_imageAvailableSemaphores[i]);
    vkCreateSemaphore(m_device.getHandle(), &semaphoreInfo, nullptr,
                      &m_renderFinishedSemaphores[i]);
    vkCreateFence(m_device.getHandle(), &fenceInfo, nullptr,
                  &m_inFlightFences[i]);
  }
}

VkResult VulkanSwapchain::acquireNextImage(uint32_t currentFrameIndex,
                                           uint32_t &imageIndex) {
  // 1. 等待 CPU 侧的 Fence，确保这一帧的资源已经不再被 GPU 使用
  vkWaitForFences(m_device.getHandle(), 1,
                  &m_inFlightFences[currentFrameIndex], VK_TRUE, UINT64_MAX);

  // 2. 从交换链请求图像
  return vkAcquireNextImageKHR(
      m_device.getHandle(), m_handle, UINT64_MAX,
      m_imageAvailableSemaphores[currentFrameIndex], // 图像可用时发出的信号
      VK_NULL_HANDLE, &imageIndex);
}

VkResult VulkanSwapchain::present(uint32_t currentFrameIndex,
                                  uint32_t imageIndex) {
  VkPresentInfoKHR presentInfo{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};

  // 等待渲染完成信号量
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &m_renderFinishedSemaphores[currentFrameIndex];

  VkSwapchainKHR swapChains[] = {m_handle};
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = swapChains;
  presentInfo.pImageIndices = &imageIndex;

  return vkQueuePresentKHR(m_device.getPresentQueue(), &presentInfo);
}

void VulkanSwapchain::rebuild(VkExtent2D newExtent,
                              VulkanRenderPass &renderPass) {
  vkDeviceWaitIdle(m_device.getHandle());

  cleanup(); // 销毁旧资源

  createInternal(newExtent);
  createImageViews();
  createDepthResources();
  // 注意：同步对象不需要随窗口重建而重建，这里假设它们已存在
  setupFramebuffers(renderPass);
}

// 辅助方法（示例中未包含 Depth 具体创建逻辑，因为依赖 Device 的内存分配封装）
void VulkanSwapchain::createDepthResources() {
  // 实际实现中需要调用 vkCreateImage 和渲染后端特有的内存分配逻辑
}

void VulkanSwapchain::setupFramebuffers(VulkanRenderPass &renderPass) {
  m_framebuffers.clear();
  for (auto imageView : m_imageViews) {
    // 伪代码：构造函数应根据你的 VulkanFrameBuffer 定义调整
    m_framebuffers.push_back(std::make_unique<VulkanFrameBuffer>(
        m_device, renderPass, m_extent, imageView, m_depthImageView));
  }
}

// 获取接口实现...
VkSemaphore VulkanSwapchain::getImageAvailableSemaphore(uint32_t i) const {
  return m_imageAvailableSemaphores[i];
}
VkSemaphore VulkanSwapchain::getRenderFinishedSemaphore(uint32_t i) const {
  return m_renderFinishedSemaphores[i];
}
VkFence VulkanSwapchain::getInFlightFence(uint32_t i) const {
  return m_inFlightFences[i];
}

} // namespace LX_core::graphic_backend