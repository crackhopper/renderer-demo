#include "core/scene/frame_graph.hpp"

#include "core/scene/scene.hpp"
#include <unordered_set>

namespace LX_core {

void FrameGraph::addPass(FramePass pass) {
  m_passes.push_back(std::move(pass));
}

void FrameGraph::buildFromScene(const Scene &scene) {
  for (auto &pass : m_passes) {
    pass.queue.clearItems();
    for (const auto &renderable : scene.getRenderables()) {
      if (!renderable)
        continue;
      RenderingItem item =
          scene.buildRenderingItemForRenderable(renderable, pass.name);
      pass.queue.addItem(std::move(item));
    }
    pass.queue.sort();
  }
}

std::vector<PipelineBuildInfo>
FrameGraph::collectAllPipelineBuildInfos() const {
  std::unordered_set<PipelineKey, PipelineKey::Hash> seen;
  std::vector<PipelineBuildInfo> out;
  for (const auto &pass : m_passes) {
    for (auto info : pass.queue.collectUniquePipelineBuildInfos()) {
      if (seen.insert(info.key).second)
        out.push_back(std::move(info));
    }
  }
  return out;
}

} // namespace LX_core
