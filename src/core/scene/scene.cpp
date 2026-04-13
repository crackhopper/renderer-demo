#include "scene.hpp"
#include "core/resources/mesh.hpp"

namespace LX_core {

RenderingItem
Scene::buildRenderingItemForRenderable(const IRenderablePtr &renderable,
                                       StringID pass) const {
  RenderingItem item;
  if (!renderable)
    return item;

  item.vertexBuffer = renderable->getVertexBuffer();
  item.indexBuffer = renderable->getIndexBuffer();
  item.objectInfo = renderable->getObjectInfo();
  item.descriptorResources = renderable->getDescriptorResources();
  item.shaderInfo = renderable->getShaderInfo();
  item.passMask = renderable->getPassMask();
  item.pass = pass;

  auto sub = std::dynamic_pointer_cast<RenderableSubMesh>(renderable);
  if (sub && sub->mesh && sub->material) {
    item.material = sub->material;
    StringID objectSig = sub->getRenderSignature(pass);
    StringID materialSig = sub->material->getRenderSignature(pass);
    item.pipelineKey = PipelineKey::build(objectSig, materialSig);
  }
  return item;
}

RenderingItem Scene::buildRenderingItem(StringID pass) {
  if (m_renderables.empty())
    return {};
  return buildRenderingItemForRenderable(m_renderables.front(), pass);
}

} // namespace LX_core
