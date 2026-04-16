#pragma once

#include "core/asset/shader.hpp"
#include "core/utils/string_table.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>

namespace LX_core {

enum class CullMode : uint8_t { None, Front, Back };
enum class CompareOp : uint8_t { Less, LessEqual, Greater, Equal, Always };
enum class BlendFactor : uint8_t { Zero, One, SrcAlpha, OneMinusSrcAlpha };

inline const char *toString(CullMode m) {
  switch (m) {
  case CullMode::None:
    return "CullNone";
  case CullMode::Front:
    return "CullFront";
  case CullMode::Back:
    return "CullBack";
  }
  return "CullUnknown";
}

inline const char *toString(CompareOp op) {
  switch (op) {
  case CompareOp::Less:
    return "Less";
  case CompareOp::LessEqual:
    return "LessEqual";
  case CompareOp::Greater:
    return "Greater";
  case CompareOp::Equal:
    return "Equal";
  case CompareOp::Always:
    return "Always";
  }
  return "CmpUnknown";
}

inline const char *toString(BlendFactor f) {
  switch (f) {
  case BlendFactor::Zero:
    return "Zero";
  case BlendFactor::One:
    return "One";
  case BlendFactor::SrcAlpha:
    return "SrcAlpha";
  case BlendFactor::OneMinusSrcAlpha:
    return "OneMinusSrcAlpha";
  }
  return "BlendUnknown";
}

struct RenderState {
  CullMode cullMode = CullMode::Back;
  bool depthTestEnable = true;
  bool depthWriteEnable = true;
  CompareOp depthOp = CompareOp::LessEqual;
  bool blendEnable = false;
  BlendFactor srcBlend = BlendFactor::One;
  BlendFactor dstBlend = BlendFactor::Zero;

  bool operator==(const RenderState &rhs) const {
    return cullMode == rhs.cullMode && depthTestEnable == rhs.depthTestEnable &&
           depthWriteEnable == rhs.depthWriteEnable && depthOp == rhs.depthOp &&
           blendEnable == rhs.blendEnable && srcBlend == rhs.srcBlend &&
           dstBlend == rhs.dstBlend;
  }

  size_t getHash() const {
    size_t h = 0;
    hash_combine(h, static_cast<uint32_t>(cullMode));
    hash_combine(h, depthTestEnable);
    hash_combine(h, depthWriteEnable);
    hash_combine(h, static_cast<uint32_t>(depthOp));
    hash_combine(h, blendEnable);
    hash_combine(h, static_cast<uint32_t>(srcBlend));
    hash_combine(h, static_cast<uint32_t>(dstBlend));
    return h;
  }

  StringID getRenderSignature() const {
    auto &tbl = GlobalStringTable::get();
    StringID fields[] = {
        tbl.Intern(toString(cullMode)),
        tbl.Intern(depthTestEnable ? "DepthTest" : "NoDepthTest"),
        tbl.Intern(depthWriteEnable ? "DepthWrite" : "NoDepthWrite"),
        tbl.Intern(toString(depthOp)),
        tbl.Intern(blendEnable ? "Blend" : "NoBlend"),
        tbl.Intern(toString(srcBlend)),
        tbl.Intern(toString(dstBlend)),
    };
    return tbl.compose(TypeTag::RenderState, fields);
  }
};

struct MaterialPassDefinition {
  RenderState renderState;
  ShaderProgramSet shaderSet;
  std::unordered_map<std::string, ShaderResourceBinding> bindingCache;

  size_t getHash() const {
    size_t h = renderState.getHash();
    hash_combine(h, shaderSet.getHash());
    return h;
  }

  StringID getRenderSignature() const {
    StringID fields[] = {
        shaderSet.getRenderSignature(),
        renderState.getRenderSignature(),
    };
    return GlobalStringTable::get().compose(TypeTag::MaterialPassDefinition,
                                            fields);
  }

  std::optional<std::reference_wrapper<const ShaderResourceBinding>>
  findBinding(const std::string &name) const {
    auto it = bindingCache.find(name);
    if (it != bindingCache.end())
      return it->second;
    return std::nullopt;
  }

  void buildCache() {
    bindingCache.clear();
    auto shader = shaderSet.getShader();
    if (!shader)
      return;

    for (const auto &binding : shader->getReflectionBindings()) {
      bindingCache[binding.name] = binding;
    }
  }
};

} // namespace LX_core
