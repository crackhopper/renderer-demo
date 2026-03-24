## MODIFIED Requirements

### Requirement: VulkanSwapchain shall manage window-presentation synchronization

The VulkanSwapchain SHALL support:
- Querying surface capabilities and selecting appropriate image count
- Creating swapchain with VK_PRESENT_MODE_FIFO_KHR (vsync)
- Acquiring next image with semaphore for render synchronization
- Presenting rendered image to window
- Using device-discovered surface format for swapchain images
- Using device-discovered depth format for depth resources

#### Scenario: Swapchain initialization
- **WHEN** Creating swapchain for window 800x600
- **THEN** Swapchain SHALL have minImageCount >= 2 and extent 800x600

#### Scenario: Swapchain uses device-selected surface format
- **WHEN** Creating swapchain with VulkanDevice
- **THEN** Swapchain SHALL use the surface format (VkSurfaceFormatKHR) returned by device->getSurfaceFormat()

#### Scenario: Depth resource creation uses device-selected depth format
- **WHEN** Creating depth resources for swapchain
- **THEN** Depth image SHALL be created with the format returned by device->getDepthFormat()
- **AND** SHALL use VK_IMAGE_TILING_OPTIMAL and VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
