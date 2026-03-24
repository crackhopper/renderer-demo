## 1. VulkanSwapchain Format Fixes

- [x] 1.1 In `vkr_swapchain.cpp` `createInternal()`, replace hardcoded `VK_FORMAT_B8G8R8A8_SRGB` and `VK_COLOR_SPACE_SRGB_NONLINEAR_KHR` with `m_device.getSurfaceFormat().format` and `m_device.getSurfaceFormat().colorSpace`
- [x] 1.2 In `vkr_swapchain.cpp` `initialize()`, replace `m_depthFormat = renderPass.getDepthFormat()` with `m_depthFormat = m_device.getDepthFormat()`
- [x] 1.3 In `vkr_swapchain.cpp` `rebuild()`, replace `m_depthFormat = renderPass.getDepthFormat()` with `m_depthFormat = m_device.getDepthFormat()`
- [x] 1.4 Update `getImageFormat()` in `vkr_swapchain.hpp` to return `m_device.getSurfaceFormat().format` instead of the stored `m_imageFormat` member

## 2. Verification

- [x] 2.1 Build the project to verify no compilation errors (swapchain changes compiled successfully; pre-existing test file errors unrelated to this change)
- [x] 2.2 Run render tests to verify functionality
