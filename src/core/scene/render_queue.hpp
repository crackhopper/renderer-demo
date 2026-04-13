#pragma once

#include "core/resources/pipeline_build_info.hpp"
#include "core/scene/scene.hpp"
#include <vector>

namespace LX_core {

/// 一个 pass 内按 PipelineKey 聚合的 RenderingItem 队列。
/// FrameGraph::buildFromScene 填充每个 FramePass 的 RenderQueue，
/// 预构建时调用 collectUniquePipelineBuildInfos 去重后交给 backend。
class RenderQueue {
public:
  void addItem(RenderingItem item);
  void clearItems();

  /// 稳定排序：相同 pipelineKey 的项相邻，降低 pipeline 切换开销。
  void sort();

  const std::vector<RenderingItem> &getItems() const { return m_items; }

  /// 按 PipelineKey 去重后返回 PipelineBuildInfo 列表。
  std::vector<PipelineBuildInfo> collectUniquePipelineBuildInfos() const;

private:
  std::vector<RenderingItem> m_items;
};

} // namespace LX_core
