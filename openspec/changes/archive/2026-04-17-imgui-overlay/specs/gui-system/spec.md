## MODIFIED Requirements

### Requirement: GUI initialization with Vulkan

The GUI system SHALL initialize Dear ImGui with Vulkan rendering backend using SDL3 for input. `Gui::InitParams` SHALL be:

```cpp
struct InitParams {
  VkInstance instance;
  VkPhysicalDevice physicalDevice;
  VkDevice device;
  uint32_t graphicsQueueFamilyIndex;
  uint32_t presentQueueFamilyIndex;
  VkQueue graphicsQueue;
  VkQueue presentQueue;
  VkSurfaceKHR surface;
  void* nativeWindowHandle;
  VkRenderPass renderPass;
  uint32_t swapchainImageCount;
};
```

`Gui::init()` SHALL execute, in order:

1. `IMGUI_CHECKVERSION()`
2. `ImGui::CreateContext()`
3. `ImGui::StyleColorsDark()`
4. `ImGui_ImplSDL3_InitForVulkan(static_cast<SDL_Window*>(params.nativeWindowHandle))`
5. Create an ImGui-owned `VkDescriptorPool` sized for ImGui (at least 1000 `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER`, `VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT`)
6. `ImGui_ImplVulkan_Init(...)` with `RenderPass = params.renderPass`, `MinImageCount/ImageCount = params.swapchainImageCount`, `DescriptorPool` set to the pool created in step 5

The descriptor pool SHALL be owned by `Gui` and survive from `init()` to `shutdown()`.

#### Scenario: Initialize ImGui with Vulkan

- **WHEN** `Gui::init()` is called with a valid `InitParams` whose `renderPass` and `swapchainImageCount` match a live Vulkan renderer
- **THEN** ImGui context SHALL be created, SDL3 backend and Vulkan backend SHALL both be initialized, and a Gui-owned descriptor pool SHALL be live

#### Scenario: Initialize ImGui twice

- **WHEN** `init()` is called while `isInitialized()` is `true`
- **THEN** a `std::runtime_error` SHALL be thrown indicating already initialized

### Requirement: GUI frame management

The GUI system SHALL provide `beginFrame()` and `endFrame(VkCommandBuffer)` methods to delimit ImGui frame rendering. `endFrame` SHALL accept the current command buffer from the caller and SHALL NOT silently use `VK_NULL_HANDLE`.

`beginFrame()` SHALL call, in order:

1. `ImGui_ImplSDL3_NewFrame()`
2. `ImGui_ImplVulkan_NewFrame()`
3. `ImGui::NewFrame()`

`endFrame(VkCommandBuffer cmd)` SHALL call `ImGui::Render()` then pass the resulting `ImDrawData*` to `ImGui_ImplVulkan_RenderDrawData(drawData, cmd, VK_NULL_HANDLE)`. If `drawData` is null or `drawData->TotalVtxCount == 0`, the method SHALL return without submitting.

#### Scenario: Start new ImGui frame

- **WHEN** `beginFrame()` is called after initialization
- **THEN** SDL3 `NewFrame`, Vulkan `NewFrame`, and `ImGui::NewFrame` SHALL have been called in that order

#### Scenario: End and render ImGui frame with real command buffer

- **WHEN** `endFrame(cmd)` is called with a valid recording command buffer and at least one ImGui widget was emitted
- **THEN** `ImGui::Render()` SHALL be called and `ImGui_ImplVulkan_RenderDrawData(drawData, cmd, VK_NULL_HANDLE)` SHALL be invoked

#### Scenario: endFrame skips empty draw data

- **WHEN** `endFrame(cmd)` is called in a frame where no ImGui widgets were emitted
- **THEN** `ImGui_ImplVulkan_RenderDrawData` SHALL NOT be invoked and the call SHALL return without error

### Requirement: GUI shutdown

The GUI system SHALL properly shut down ImGui and release resources in reverse order of initialization:

1. `ImGui_ImplVulkan_Shutdown()`
2. `ImGui_ImplSDL3_Shutdown()`
3. Destroy the Gui-owned descriptor pool via `vkDestroyDescriptorPool`
4. `ImGui::DestroyContext()`

#### Scenario: Shutdown ImGui

- **WHEN** `shutdown()` is called after a successful `init()`
- **THEN** `ImGui_ImplVulkan_Shutdown()`, `ImGui_ImplSDL3_Shutdown()`, descriptor pool destruction, and `ImGui::DestroyContext()` SHALL all occur before the method returns; `isInitialized()` SHALL return `false` afterwards
