#include "core/resources/pipeline_build_info.hpp"

#include "core/resources/mesh.hpp"
#include "core/scene/scene.hpp"

#include <cassert>

namespace LX_core {

PipelineBuildInfo
PipelineBuildInfo::fromRenderingItem(const RenderingItem &item) {
  assert(item.shaderInfo &&
         "PipelineBuildInfo::fromRenderingItem: shaderInfo required");
  assert(item.vertexBuffer &&
         "PipelineBuildInfo::fromRenderingItem: vertexBuffer required");
  assert(item.indexBuffer &&
         "PipelineBuildInfo::fromRenderingItem: indexBuffer required");
  assert(item.material &&
         "PipelineBuildInfo::fromRenderingItem: material required");

  PipelineBuildInfo info;
  info.key = item.pipelineKey;
  info.stages = item.shaderInfo->getAllStages();
  info.bindings = item.shaderInfo->getReflectionBindings();

  auto vb = std::dynamic_pointer_cast<IVertexBuffer>(item.vertexBuffer);
  assert(vb && "PipelineBuildInfo::fromRenderingItem: vertex buffer is not "
               "IVertexBuffer");
  info.vertexLayout = vb->getLayout();

  auto ib = std::dynamic_pointer_cast<IndexBuffer>(item.indexBuffer);
  assert(
      ib &&
      "PipelineBuildInfo::fromRenderingItem: index buffer is not IndexBuffer");
  info.topology = ib->getTopology();

  info.renderState = item.material->getRenderState();

  // Engine-wide push constant convention until shader-declared ranges arrive.
  info.pushConstant = PushConstantRange{};
  return info;
}

} // namespace LX_core
