## Why

The `VulkanDevice` properly selects optimal surface and depth formats during initialization via `getSurfaceFormat()` and `getDepthFormat()`. However, `VulkanSwapchain` ignores these device-selected formats and hardcodes its own:
- Hardcodes `VK_FORMAT_B8G8R8A8_SRGB` for swapchain image format (line 90-91 in `vkr_swapchain.cpp`)
- Gets depth format from `VulkanRenderPass` instead of directly from the device

This inconsistency can lead to format mismatches and suboptimal GPU performance when the device's preferred formats differ from the hardcoded values.

## What Changes

1. Modify `VulkanSwapchain::createInternal` to use `m_device.getSurfaceFormat()` instead of hardcoded `VK_FORMAT_B8G8R8A8_SRGB`
2. Modify `VulkanSwapchain::initialize` and `rebuild` to use `m_device.getDepthFormat()` instead of `renderPass.getDepthFormat()`
3. Remove redundant format members if no longer needed

## Capabilities

### New Capabilities
- None - this is a refactoring/fix within existing behavior

### Modified Capabilities
- `renderer-backend-vulkan`: VulkanSwapchain format selection - change from hardcoded values to device-discovered formats

## Impact

- **Affected Files**: `vkr_swapchain.cpp`, `vkr_swapchain.hpp`
- **Risk**: Low - this aligns existing code with the device's chosen formats
- **Testing**: Existing render tests should continue to pass; format consistency verified
