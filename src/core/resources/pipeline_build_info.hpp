#pragma once

#include "core/resources/index_buffer.hpp"
#include "core/resources/material.hpp"
#include "core/resources/pipeline_key.hpp"
#include "core/resources/shader.hpp"
#include "core/resources/vertex_buffer.hpp"
#include <cstdint>
#include <vector>

namespace LX_core {

struct RenderingItem; // forward decl

/// Backend-agnostic push constant declaration. Engine-wide convention today is
/// a 128-byte range visible to vertex + fragment stages; future shader-declared
/// ranges can extend this struct without touching backend-neutral code.
struct PushConstantRange {
  uint32_t offset = 0;
  uint32_t size = 128;
  /// Bitmask of ShaderStage values that need access. Default: Vertex |
  /// Fragment.
  uint32_t stageFlagsMask = static_cast<uint32_t>(ShaderStage::Vertex) |
                            static_cast<uint32_t>(ShaderStage::Fragment);
};

/// Aggregates every input a backend needs to construct a graphics pipeline.
/// Single source of truth for "what does this pipeline need" — built from a
/// RenderingItem via `fromRenderingItem`; backends translate its fields to
/// their own types (e.g., VkVertexInputAttributeDescription).
struct PipelineBuildInfo {
  PipelineKey key;
  std::vector<ShaderStageCode> stages;
  std::vector<ShaderResourceBinding> bindings;
  VertexLayout vertexLayout;
  RenderState renderState;
  PrimitiveTopology topology = PrimitiveTopology::TriangleList;
  PushConstantRange pushConstant;

  /// Derive a complete PipelineBuildInfo from a fully-built RenderingItem.
  /// Requires `item.shaderInfo`, `item.vertexBuffer`, `item.indexBuffer`, and
  /// `item.material` to be non-null.
  static PipelineBuildInfo fromRenderingItem(const RenderingItem &item);
};

} // namespace LX_core
