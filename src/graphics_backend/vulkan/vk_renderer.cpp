#include "vk_renderer.hpp"
#include "details/commands/vkc_cmdbuffer_manager.hpp"
#include "details/render_objects/vkr_rendercontext.hpp"
#include "details/render_objects/vkr_swapchain.hpp"
#include "details/vk_device.hpp"
#include "details/vk_resource_manager.hpp"
#include "infra/window/window.hpp"
#include <cstring>
#include <stdexcept>
namespace {
// 这种函数通常只需要执行一次，逻辑相对固定
VkFormat findSupportedFormat(VkPhysicalDevice physicalDevice,
                             const std::vector<VkFormat> &candidates,
                             VkImageTiling tiling,
                             VkFormatFeatureFlags features) {
  for (VkFormat format : candidates) {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
    if (tiling == VK_IMAGE_TILING_LINEAR &&
        (props.linearTilingFeatures & features) == features)
      return format;
    else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
             (props.optimalTilingFeatures & features) == features)
      return format;
  }
  throw std::runtime_error("failed to find supported format!");
}
} // namespace

namespace LX_core::graphic_backend {

class VulkanRendererImpl : public gpu::Renderer {
public:
  VulkanRendererImpl() {}
  ~VulkanRendererImpl() override { destroy(); }

  void initialize(WindowPtr _window) override {
    window = _window;
    device = std::make_unique<VulkanDevice>();
    device->initialize();
    surface = (VkSurfaceKHR)window->createGraphicsHandle(GraphicsAPI::Vulkan,
                                                         device->getInstance());
    if (surface == nullptr) {
      throw std::runtime_error("Failed to create Vulkan surface");
    }
    surfaceFormat = findBestSurfaceFormat(device->getPhysicalDevice(), surface);
    depthFormat = findSupportedFormat(
        device->getPhysicalDevice(), {VK_FORMAT_D32_SFLOAT_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);

    VkExtent2D extent = {window->getWidth(), window->getHeight()};
    const int maxFramesInFlight = 3;

    resourceManager = VulkanResourceManager::create(*device);
    resourceManager->initializeRenderPassAndPipeline(surfaceFormat,
                                                     depthFormat);

    auto graphicsIdx = device->getGraphicsQueueFamilyIndex(); 
    auto presentIdx = device->getPresentQueueFamilyIndex(); 
    swapchain =
        VulkanSwapchain::create(*device, surface, extent, graphicsIdx, presentIdx, maxFramesInFlight);
    swapchain->initialize(resourceManager->getRenderPass());

    cmdBufferMgr = VulkanCommandBufferManager::create(
        *device, maxFramesInFlight, device->getGraphicsQueueFamilyIndex());
  }
  void shutdown() override {}
  void initScene(ScenePtr scene) override {}

  void uploadData() override {}
  void draw() override {}

  WindowPtr window = nullptr;
  VkSurfaceKHR surface = nullptr;
  VkSurfaceFormatKHR surfaceFormat = {};
  VkFormat depthFormat = {};

  VulkanDevicePtr device = nullptr;
  VulkanResourceManagerPtr resourceManager = nullptr;
  VulkanSwapchainPtr swapchain = nullptr;
  VulkanCommandBufferManagerPtr cmdBufferMgr = nullptr;

private:
  void destroy() {
    if (device) {
      // 关键：等 GPU 干完活再删东西
      vkDeviceWaitIdle(device->getHandle());
    }
    // 1. 销毁 Command Buffer Manager
    cmdBufferMgr.reset();
    // 2. 销毁 Swapchain
    swapchain.reset();
    // 3. 销毁 Resource Manager
    resourceManager.reset();
    // 4. 销毁 Surface
    window->destroyGraphicsHandle(GraphicsAPI::Vulkan, device->getInstance(),
                                  surface);
    surface = VK_NULL_HANDLE;
    // 5. 销毁 Device
    device.reset();
  }

  VkSurfaceFormatKHR findBestSurfaceFormat(VkPhysicalDevice physicalDevice,
                                           VkSurfaceKHR surface) {
    // 1. 获取硬件支持的所有表面格式
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount,
                                         nullptr);

    if (formatCount == 0) {
      throw std::runtime_error("No surface formats found!");
    }

    std::vector<VkSurfaceFormatKHR> availableFormats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount,
                                         availableFormats.data());

    // 2. 筛选最优格式
    for (const auto &availableFormat : availableFormats) {
      // 我们优先寻找 B8G8R8A8 或 R8G8B8A8 的 SRGB 非线性版本
      // SRGB 可以提供更准确的视觉亮度（Gamma 校正）
      if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
          availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        return availableFormat;
      }
    }

    // 3. 兜底方案：如果找不到 SRGB，直接返回第一个支持的格式
    return availableFormats[0];
  }

  VkFormat findDepthFormat() {
    return findSupportedFormat(device->getPhysicalDevice(),
                               {VK_FORMAT_D32_SFLOAT,
                                VK_FORMAT_D32_SFLOAT_S8_UINT,
                                VK_FORMAT_D24_UNORM_S8_UINT},
                               VK_IMAGE_TILING_OPTIMAL,
                               VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
  }

  VkSurfaceFormatKHR chooseSwapSurfaceFormat(
      const std::vector<VkSurfaceFormatKHR> &availableFormats) {
    // 优先选择 SRGB 非线性格式
    for (const auto &availableFormat : availableFormats) {
      if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
          availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        return availableFormat;
      }
    }
    return availableFormats[0];
  }
};

VulkanRenderer::VulkanRenderer(Token token) : p_impl(nullptr) {
  p_impl = new VulkanRendererImpl();
}

VulkanRenderer::~VulkanRenderer() { delete p_impl; }

} // namespace LX_core::graphic_backend
