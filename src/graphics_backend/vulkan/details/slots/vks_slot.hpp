#pragma once
#include "core/resources/schema.hpp"

namespace LX_core::graphic_backend {

enum class SlotBindingStage : u8 {
  BS_VERTEX = 1,
  BS_FRAGMENT = 2,
  BS_ALL = BS_VERTEX | BS_FRAGMENT,
};

struct ResourceSlotDetails {
  SlotId semantic;
  SlotType type;
  SlotBindingStage stage;
  u32 setIndex;
  u32 binding;
  usize size;
};  
}