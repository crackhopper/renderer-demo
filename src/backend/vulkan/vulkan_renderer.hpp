#pragma once
#include "core/rhi/renderer.hpp"
#include <functional>
#include <vector>

namespace LX_core::backend {
class VulkanRendererImpl;
class VulkanRenderer;
using VulkanRendererPtr = std::unique_ptr<VulkanRenderer>;
class VulkanRenderer : public gpu::Renderer {
public:
  struct Token {};
  explicit VulkanRenderer(Token token);
  ~VulkanRenderer() override;
  static VulkanRendererPtr create(Token token){
    return std::make_unique<VulkanRenderer>(token);
  }

  void initialize(WindowPtr window, const char *appName) override;
  void shutdown() override;
  void initScene(ScenePtr scene) override;

  void uploadData() override;
  void draw() override;

  // Register a callback invoked every frame inside the swapchain render pass,
  // between Gui::beginFrame() and scene draw calls. Replace semantics; pass
  // an empty std::function to clear. Not lifted to the gpu::Renderer base.
  void setDrawUiCallback(std::function<void()> cb);

private:
  VulkanRendererImpl* p_impl = nullptr;
};

} // namespace LX_core::backend