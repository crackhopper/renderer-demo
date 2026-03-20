#pragma once
#include "core/gpu/render_resource.hpp"

namespace LX_core::graphic_backend {

enum class PipelineSlotStage : u8 {
  NONE = 0,
  VERTEX = 1,
  FRAGMENT = 2,
  ALL = VERTEX | FRAGMENT,
};


struct PipelineSlotDetails {
  PipelineSlotId id = PipelineSlotId::None;
  ResourceType type = ResourceType::None;
  PipelineSlotStage stage = PipelineSlotStage::NONE;
  u32 setIndex = 0;
  u32 binding = 0;
  usize size = 0;
};



}