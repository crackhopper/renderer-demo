#pragma once
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

namespace LX_core::graphic_backend {

// 前置声明
class VulkanDescriptorManager;
class VulkanCommandBufferManager;

/**
 * @brief Vulkan 逻辑设备包装类
 * 负责物理设备选择、逻辑设备创建、全局队列获取以及核心管理器的生命周期。
 */
class VulkanDevice {
  struct Token {};

public:
  // 强制通过工厂方法创建
  explicit VulkanDevice(Token);
  ~VulkanDevice();

  static std::unique_ptr<VulkanDevice> create() {
    return std::make_unique<VulkanDevice>(Token{});
  }

  // --- 生命周期 ---
  void initialize();
  void shutdown();

  // --- 句柄获取 (只读) ---
  VkDevice getHandle() const { return m_device; }
  VkPhysicalDevice getPhysicalDevice() const { return m_physicalDevice; }
  VkInstance getInstance() const { return m_instance; }
  VkQueue getGraphicsQueue() const { return m_graphicsQueue; }
  VkQueue getPresentQueue() const { return m_presentQueue; }

  uint32_t getGraphicsQueueFamilyIndex() const {
    return m_graphicsQueueFamilyIndex;
  }
  uint32_t getPresentQueueFamilyIndex() const {
    return m_presentQueueFamilyIndex;
  }

  // --- 核心管理器访问 ---

  /**
   * @brief 描述符管理器：负责全局 Layout 缓存和每帧 Set 分配
   */
  VulkanDescriptorManager &getDescriptorManager() {
    return *m_descriptorManager;
  }

  // --- 实用工具 ---

  /**
   * @brief 查找内存类型索引 (用于 Buffer/Image 分配)
   */
  uint32_t findMemoryTypeIndex(uint32_t typeFilter,
                               VkMemoryPropertyFlags properties) const;

  /**
   * @brief 辅助函数：等待设备空闲（通常在 shutdown 或重建 Swapchain 前调用）
   */
  void waitIdle() const { vkDeviceWaitIdle(m_device); }

private:
  // 内部初始化流程
  void createInstance();
  void pickPhysicalDevice();
  void createLogicalDevice();

private:
  // 管理器：由 Device 持有，因为它们的生命周期与 Device 一致
  std::unique_ptr<VulkanDescriptorManager> m_descriptorManager;

  // Vulkan 核心句柄
  VkInstance m_instance = VK_NULL_HANDLE;
  VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
  VkDevice m_device = VK_NULL_HANDLE;

  // 队列
  VkQueue m_graphicsQueue = VK_NULL_HANDLE;
  VkQueue m_presentQueue = VK_NULL_HANDLE;
  uint32_t m_graphicsQueueFamilyIndex = 0;
  uint32_t m_presentQueueFamilyIndex = 0;

  // 调试层
  VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
};

using VulkanDevicePtr = std::unique_ptr<VulkanDevice>;

} // namespace LX_core::graphic_backend