#pragma once

#include "core/asset/texture.hpp"

#include <string>

namespace LX_infra {

/// Lazily-created singleton placeholder textures.
LX_core::CombinedTextureSamplerPtr getPlaceholderWhite();
LX_core::CombinedTextureSamplerPtr getPlaceholderBlack();
LX_core::CombinedTextureSamplerPtr getPlaceholderNormal();

/// Resolve a placeholder name ("white", "black", "normal") to the
/// corresponding texture, or nullptr if the name is not a placeholder.
LX_core::CombinedTextureSamplerPtr
resolvePlaceholder(const std::string &name);

} // namespace LX_infra
