#pragma once
#include "core/resources/pipeline_key.hpp"
#include "core/resources/shader.hpp"
#include "core/scene/camera.hpp"
#include "core/scene/light.hpp"
#include "core/scene/object.hpp"
#include "core/scene/pass.hpp"

namespace LX_core {

using ShaderPtr = IShaderPtr;

// 简化 RenderingItem
struct RenderingItem {
  ShaderPtr shaderInfo;
  MaterialPtr material; // 材质句柄 — 用于 PipelineBuildInfo::fromRenderingItem

  ObjectPCPtr objectInfo;
  IRenderResourcePtr vertexBuffer;
  IRenderResourcePtr indexBuffer;

  std::vector<IRenderResourcePtr> descriptorResources; // 材质 + skeleton 等资源

  ResourcePassFlag passMask;
  StringID pass;
  PipelineKey pipelineKey;
};

// Scene 层简化示例
class Scene {
public:
  using Ptr = std::shared_ptr<Scene>;

  CameraPtr camera;
  DirectionalLightPtr directionalLight;

  Scene(IRenderablePtr mesh) {
    if (mesh)
      m_renderables.push_back(std::move(mesh));
    camera = std::make_shared<Camera>(ResourcePassFlag::Forward);
    directionalLight =
        std::make_shared<DirectionalLight>(ResourcePassFlag::Forward);
  }

  static auto create(IRenderablePtr mesh) {
    return std::make_shared<Scene>(mesh);
  }

  const std::vector<IRenderablePtr> &getRenderables() const {
    return m_renderables;
  }

  void addRenderable(IRenderablePtr r) {
    m_renderables.push_back(std::move(r));
  }

  RenderingItem buildRenderingItem(StringID pass);

  /// 为一个具体 renderable 构造 RenderingItem。供 FrameGraph 迭代使用。
  RenderingItem
  buildRenderingItemForRenderable(const IRenderablePtr &renderable,
                                  StringID pass) const;

private:
  std::vector<IRenderablePtr> m_renderables;
};

using ScenePtr = Scene::Ptr;
} // namespace LX_core
