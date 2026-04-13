#pragma once

#include "core/gpu/image_format.hpp"
#include <cstddef>
#include <cstdint>

namespace LX_core {

/// Describes a render pass attachment set (color + depth + MSAA). Not part of
/// PipelineKey today; reserved for future multi-target support.
struct RenderTarget {
  ImageFormat colorFormat = ImageFormat::BGRA8;
  ImageFormat depthFormat = ImageFormat::D32Float;
  uint8_t sampleCount = 1;

  size_t getHash() const;
};

} // namespace LX_core
