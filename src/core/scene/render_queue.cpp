#include "core/scene/render_queue.hpp"

#include <algorithm>
#include <unordered_set>

namespace LX_core {

void RenderQueue::addItem(RenderingItem item) {
  m_items.push_back(std::move(item));
}

void RenderQueue::clearItems() { m_items.clear(); }

void RenderQueue::sort() {
  std::stable_sort(m_items.begin(), m_items.end(),
                   [](const RenderingItem &a, const RenderingItem &b) {
                     return a.pipelineKey.id.id < b.pipelineKey.id.id;
                   });
}

std::vector<PipelineBuildInfo>
RenderQueue::collectUniquePipelineBuildInfos() const {
  std::unordered_set<PipelineKey, PipelineKey::Hash> seen;
  std::vector<PipelineBuildInfo> out;
  out.reserve(m_items.size());
  for (const auto &item : m_items) {
    if (!seen.insert(item.pipelineKey).second)
      continue;
    out.push_back(PipelineBuildInfo::fromRenderingItem(item));
  }
  return out;
}

} // namespace LX_core
