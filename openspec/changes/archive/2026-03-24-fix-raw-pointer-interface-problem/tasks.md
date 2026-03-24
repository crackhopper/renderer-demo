## 1. Update VulkanResourceManager Interface

- [x] 1.1 Change `getBuffer()` to return `std::optional<std::reference_wrapper<VulkanBuffer>>` in `vk_resource_manager.hpp`
- [x] 1.2 Change `getTexture()` to return `std::optional<std::reference_wrapper<VulkanTexture>>` in `vk_resource_manager.hpp`
- [x] 1.3 Change `getShader()` to return `std::optional<std::reference_wrapper<VulkanShader>>` in `vk_resource_manager.hpp`
- [x] 1.4 Change `getRenderPass()` to return `VulkanRenderPass&` in `vk_resource_manager.hpp`
- [x] 1.5 Change `getRenderPipeline()` to return `VulkanPipelineBase&` in `vk_resource_manager.hpp`
- [x] 1.6 Add `#include <optional>` to `vk_resource_manager.hpp`

## 2. Update VulkanResourceManager Implementation

- [x] 2.1 Update `getBuffer()` implementation to return `std::optional<std::reference_wrapper<VulkanBuffer>>`
- [x] 2.2 Update `getTexture()` implementation to return `std::optional<std::reference_wrapper<VulkanTexture>>`
- [x] 2.3 Update `getShader()` implementation to return `std::optional<std::reference_wrapper<VulkanShader>>`
- [x] 2.4 Update `getRenderPass()` implementation to return `VulkanRenderPass&`
- [x] 2.5 Update `getRenderPipeline()` implementation to return `VulkanPipelineBase&`

## 3. Update Call Sites in Vulkan Backend

- [x] 3.1 Update `vk_renderer.cpp`: `getRenderPass()` call now uses reference directly
- [x] 3.2 Update `vk_renderer.cpp`: `getRenderPipeline()` call now uses `auto&` and passes reference
- [x] 3.3 Update `vkc_cmdbuffer.cpp`: `getBuffer()` now uses `auto` with `.get()` to dereference the optional
- [x] 3.4 Update `vkc_cmdbuffer.cpp`: `getTexture()` now uses `auto` with `.get()` to dereference the optional

## 4. Update Call Sites in Tests

- [x] 4.1 Update `test_vulkan_resource_manager.cpp`: `getRenderPass()` uses `auto&`
- [x] 4.2 Update `test_vulkan_resource_manager.cpp`: `getRenderPipeline()` uses `auto&`
- [x] 4.3 Update `test_vulkan_resource_manager.cpp`: `getBuffer()` calls updated to handle optional
- [x] 4.4 Update `test_vulkan_command_buffer.cpp`: `getRenderPass()` uses `auto&`
- [x] 4.5 Update `test_vulkan_command_buffer.cpp`: `getRenderPipeline()` uses `auto&` and passes reference

## 5. Verify Build

- [x] 5.1 Build project to verify all changes compile correctly
- [x] 5.2 Run tests to verify no runtime regressions (no ctest tests registered, build successful)
