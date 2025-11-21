#include "vulkan/vulkan.h"
#include <iostream>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <stdlib.h>
#endif

void expSetEnvVK() {
#ifdef _WIN32
  SetEnvironmentVariableW(
      L"VK_DRIVER_FILES",
      LR"(C:\WINDOWS\System32\DriverStore\FileRepository\nvmi.inf_amd64_c6ae241e95feb82d\nv-vk64.json)");

  SetEnvironmentVariableW(L"VK_LOADER_LAYERS_DISABLE", L"~implicit~");
#else
  setenv(
      "VK_DRIVER_FILES",
      R"(C:\WINDOWS\System32\DriverStore\FileRepository\nvmi.inf_amd64_c6ae241e95feb82d\nv-vk64.json)",
      1);
  setenv("VK_LOADER_LAYERS_DISABLE", "~implicit~", 1);
#endif
}

bool ENABLE_VALIDATION_LAYERS = false;

VKAPI_ATTR VkBool32 VKAPI_CALL expValidationDebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
    void *pUserData) {

  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

  return VK_FALSE;
}

void expPopulateDebugMessengerCreateInfo(
    VkDebugUtilsMessengerCreateInfoEXT &createInfo) {
  // 这个函数专门填充 VkDebugUtilsMessengerCreateInfoEXT 结构体
  createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  createInfo.pfnUserCallback = expValidationDebugCallback;
}

int main() {
  expSetEnvVK();

  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Create Instance Test";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  // 这两个member用来指定启用的扩展
  auto reqExtensions = std::vector<const char *>();
  // auto reqExtensions = std::vector<const char
  // *>{VK_KHR_SURFACE_EXTENSION_NAME};
  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(reqExtensions.size());
  createInfo.ppEnabledExtensionNames = reqExtensions.data();

  // 再往下的两个member实际是指定启用的layer。这里忽略了 ppEnabledLayerNames
  if (ENABLE_VALIDATION_LAYERS) {
    auto validationLayers = std::vector<const char *>();
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();

    // 如果想要调试 vkCreateInstance 调用，需要设置 pNext 指向 一个
    // VkDebugUtilsMessengerCreateInfoEXT 结构体
    expPopulateDebugMessengerCreateInfo(debugCreateInfo);
    createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT *)&debugCreateInfo;
    std::cout << "validation layers enabled" << std::endl;
  } else {
    createInfo.enabledLayerCount = 0;
  }

  // 创建vulkan对象。
  VkInstance instance;
  auto ret = vkCreateInstance(&createInfo, nullptr, &instance);
  if (ret != VK_SUCCESS) {
    std::cout << "create instance failed with code: " << ret << std::endl;
    throw std::runtime_error("failed to create instance!");
  }

  std::cout << "create instance success" << std::endl;
  return 0;
}