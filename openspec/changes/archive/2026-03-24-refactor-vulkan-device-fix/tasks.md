## 1. Update Spec Documentation

- [x] 1.1 Update openspec/specs/renderer-backend-vulkan/spec.md to reflect new VulkanDevice interface (delta spec created in openspec/changes/refactor-vulkan-device-fix/specs/)

## 2. Fix Test Files

- [x] 2.1 Fix test_vulkan_device.cpp - pass WindowPtr and appName to initialize()
- [x] 2.2 Fix test_vulkan_buffer.cpp - pass WindowPtr and appName to initialize()
- [x] 2.3 Fix test_vulkan_shader.cpp - pass WindowPtr and appName to initialize()
- [x] 2.4 Fix test_vulkan_texture.cpp - pass WindowPtr and appName to initialize()
- [x] 2.5 Fix test_vulkan_pipeline.cpp - pass WindowPtr and appName to initialize()
- [x] 2.6 Fix test_vulkan_framebuffer.cpp - pass WindowPtr and appName to initialize()
- [x] 2.7 Fix test_vulkan_renderpass.cpp - pass WindowPtr and appName to initialize()
- [x] 2.8 Fix test_vulkan_command_buffer.cpp - pass WindowPtr and appName to initialize()
- [x] 2.9 Fix test_vulkan_depth.cpp - pass WindowPtr and appName to initialize()
- [x] 2.10 Fix test_vulkan_resource_manager.cpp - pass WindowPtr and appName to initialize()

## 3. Verify Changes

- [x] 3.1 Build project to verify all test files compile
- [x] 3.2 Run linter on modified files (no errors found)
