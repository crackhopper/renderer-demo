#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <functional>

namespace infra {

class Gui {
public:
  struct InitParams {
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    uint32_t graphicsQueueFamilyIndex;
    uint32_t presentQueueFamilyIndex;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkSurfaceKHR surface;
    // SDL path: must be an SDL_Window*. GLFW path: GLFWwindow*.
    void* nativeWindowHandle;
    VkRenderPass renderPass;
    uint32_t swapchainImageCount;
  };

  Gui();
  ~Gui();

  void init(const InitParams& params);
  void beginFrame();
  void endFrame(VkCommandBuffer cmd);
  void updateSwapchainImageCount(uint32_t imageCount);
  void shutdown();

  bool isInitialized() const;

private:
  struct Impl;
  Impl* pImpl;
};

} // namespace infra
