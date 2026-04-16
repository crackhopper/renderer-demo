#pragma once

#include "core/asset/shader.hpp"

#include <optional>
#include <string_view>

namespace LX_core {

/*****************************************************************
 * System-owned binding name set (REQ-031)
 *
 * These names are engine-reserved. Bindings with these names are
 * NOT owned by MaterialInstance. The set is intentionally small
 * and fixed; expanding it requires a new requirement/spec change.
 *****************************************************************/

inline constexpr std::string_view kSystemOwnedBindings[] = {
    "CameraUBO",
    "LightUBO",
    "Bones",
};

inline bool isSystemOwnedBinding(std::string_view name) {
  for (auto sv : kSystemOwnedBindings)
    if (sv == name)
      return true;
  return false;
}

/*****************************************************************
 * Expected descriptor type for each reserved binding name.
 * Returns nullopt for non-reserved names.
 *****************************************************************/

inline std::optional<ShaderPropertyType>
getExpectedTypeForSystemBinding(std::string_view name) {
  if (name == "CameraUBO")
    return ShaderPropertyType::UniformBuffer;
  if (name == "LightUBO")
    return ShaderPropertyType::UniformBuffer;
  if (name == "Bones")
    return ShaderPropertyType::UniformBuffer;
  return std::nullopt;
}

} // namespace LX_core
