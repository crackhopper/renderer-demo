#include "scene.hpp"
#include "core/resources/mesh.hpp"

namespace LX_core {

RenderingItem Scene::buildRenderingItem(StringID pass) {
  RenderingItem item;
  item.vertexBuffer = mesh->getVertexBuffer();
  item.indexBuffer = mesh->getIndexBuffer();
  item.objectInfo = mesh->getObjectInfo();
  item.descriptorResources = mesh->getDescriptorResources();
  item.shaderInfo = mesh->getShaderInfo();
  item.passMask = mesh->getPassMask();
  item.pass = pass;

  auto sub = std::dynamic_pointer_cast<RenderableSubMesh>(mesh);
  if (sub && sub->mesh && sub->material) {
    StringID objectSig = sub->getRenderSignature(pass);
    StringID materialSig = sub->material->getRenderSignature(pass);
    item.pipelineKey = PipelineKey::build(objectSig, materialSig);
  }
  return item;
}

} // namespace LX_core
