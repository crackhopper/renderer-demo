## Context

The `VulkanDevice::initialize()` process properly queries the GPU for optimal surface and depth formats. It stores these in:
- `m_surfaceFormat` (VkSurfaceFormatKHR) - accessed via `getSurfaceFormat()`
- `m_depthFormat` (VkFormat) - accessed via `getDepthFormat()`

However, `VulkanSwapchain` does not use these device-discovered formats:

1. In `createInternal()` (line 90-91), it hardcodes:
   ```cpp
   createInfo.imageFormat = VK_FORMAT_B8G8R8A8_SRGB;
   createInfo.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
   ```
   Instead of using `m_device.getSurfaceFormat()`.

2. In `initialize()` and `rebuild()`, it gets depth format from `renderPass.getDepthFormat()` instead of directly from the device.

This creates an inconsistency where the renderpass might use device-selected depth format, but the swapchain creates depth resources with potentially different format assumptions.

## Goals / Non-Goals

**Goals:**
- Ensure swapchain uses formats discovered and selected by the device
- Eliminate hardcoded format values in favor of device-discovered values
- Maintain consistency between depth resources created in swapchain and those expected by renderpass

**Non-Goals:**
- Do not change how `VulkanDevice` selects formats (working correctly)
- Do not modify the `VulkanRenderPass` interface or creation logic
- Do not change the order of operations in renderer initialization

## Decisions

### Decision 1: Use device surface format in swapchain

**Choice**: In `VulkanSwapchain::createInternal()`, replace hardcoded `VK_FORMAT_B8G8R8A8_SRGB` with `m_device.getSurfaceFormat()`.

**Rationale**: The device already selected an optimal format considering GPU capabilities and surface requirements. Using this ensures consistency and lets the device's format selection logic be the single source of truth.

**Alternatives considered**:
- Keep hardcoded format: Rejected as it bypasses device capability checking

### Decision 2: Use device depth format directly

**Choice**: In `VulkanSwapchain::initialize()` and `rebuild()`, use `m_device.getDepthFormat()` instead of `renderPass.getDepthFormat()`.

**Rationale**: The device's depth format is already determined during device initialization. Getting it directly from the device is more direct and avoids a dependency on renderpass initialization order. The renderpass was already created with the same device format, so this maintains consistency.

**Alternatives considered**:
- Keep using renderpass depth format: Would work but adds unnecessary coupling between swapchain and renderpass initialization order

### Decision 3: Remove redundant format members

**Choice**: Remove `m_imageFormat` and `_depthFormat` member variables from `VulkanSwapchain` since we can query from device.

**Rationale**: These members stored copies of formats that can be obtained from the device. The `getImageFormat()` method can return `m_device.getSurfaceFormat().format` and the depth format can come directly from device when needed.

**Alternatives considered**:
- Keep members for caching: Unnecessary since device getters are fast and formats don't change after init

## Risks / Trade-offs

[Risk] Device format selection might differ from previous hardcoded expectations
→ **Mitigation**: The device's format selection prioritizes SRGB non-linear when available, which matches the previous hardcoded assumption. This is a low-risk alignment change.

[Risk] Removing format members might break existing getters
→ **Mitigation**: Update `getImageFormat()` to delegate to device. The depth format was never exposed via a getter (only used internally), so no API break.

## Open Questions

None - the changes are straightforward format source alignment.
