#pragma once
#include "core/asset/material_instance.hpp"

namespace LX_core {

// Transitional compatibility alias while call sites migrate to the split
// headers and clearer pass-definition naming.
using RenderPassEntry = MaterialPassDefinition;

} // namespace LX_core
