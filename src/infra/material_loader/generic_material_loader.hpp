#pragma once

#include "core/asset/material_instance.hpp"

#include <filesystem>

namespace LX_infra {

/// Load a material from a YAML material definition file (.material).
/// The file describes shader(s), variants, default parameters, default
/// resources, and per-pass overrides. Each pass can optionally specify its
/// own shader. No material-type-specific C++ code is needed.
LX_core::MaterialInstancePtr
loadGenericMaterial(const std::filesystem::path &materialPath);

} // namespace LX_infra
