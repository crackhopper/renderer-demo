// game_app.cpp
#include "core/gpu/renderer.hpp"
#include "core/resources/index_buffer.hpp"
#include "core/resources/texture.hpp"
#include "core/resources/vertex_buffer.hpp"

// backend Vulkan 实现
#include "graphics_backend/vulkan/vk_renderer.hpp"

// 窗口系统
#include "infra/window/window.hpp"

#include <memory>

using namespace LX_core;
using namespace LX_core::gpu;

int main() {
  LX_infra::WindowImpl::Initialize();
  LX_core::WindowPtr window =
      std::make_shared<LX_infra::WindowImpl>("Test Renderer", 800, 600);

  // 创建 Renderer（Vulkan 后端）
  RendererPtr renderer =
      std::make_shared<LX_core::graphic_backend::VulkanRenderer>(window);
  renderer->initialize(window);

  auto vertexBufferPtr = VertexBuffer<VertexPosNormalUvBone>::create({
      VertexPosNormalUvBone({-5.0f, 5.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
                            {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f},
                            {0, 0, 0, 0}, {1.0f, 0.0f, 0.0f, 0.0f}),
      VertexPosNormalUvBone({5.0f, 5.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
                            {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f},
                            {0, 0, 0, 0}, {1.0f, 0.0f, 0.0f, 0.0f}),
      VertexPosNormalUvBone({5.0f, -5.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
                            {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f},
                            {0, 0, 0, 0}, {1.0f, 0.0f, 0.0f, 0.0f}),
  });

  auto indexBufferPtr = IndexBuffer::create({0, 1, 2});
  auto meshPtr =
      Mesh<VertexPosNormalUvBone>::create(vertexBufferPtr, indexBufferPtr);

  auto scene = Scene::create(meshPtr);
  renderer->initScene(scene);

  // 渲染循环
  bool running = true;
  window->onClose([&running]() { running = false; });
  while (running) {
    // 设置摄像机矩阵
    Mat4f proj = Mat4f::perspective(45.0f, 800.0f / 600.0f, 0.1f, 100.0f);
    scene->camera->position = {0.0f, 0.0f, 3.0f};
    scene->camera->target ={0.0f, 0.0f, 0.0f};
    scene->camera->up=Vec3f(0.0f, 1.0f, 0.0f);

    scene->camera->updateMatrices();
    renderer->uploadData();
    renderer->draw();
  }

  // 清理
  renderer->shutdown();

  return 0;
}