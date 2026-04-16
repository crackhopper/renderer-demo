#pragma once

#include "core/asset/material_pass_definition.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace LX_core {

class MaterialTemplate : public std::enable_shared_from_this<MaterialTemplate> {
  struct Token {};

public:
  using Ptr = std::shared_ptr<MaterialTemplate>;

  MaterialTemplate(Token, std::string name, IShaderPtr shader)
      : m_name(std::move(name)), m_shader(std::move(shader)) {}

  static Ptr create(std::string name, IShaderPtr shader) {
    return std::make_shared<MaterialTemplate>(Token{}, std::move(name),
                                              std::move(shader));
  }

  IShaderPtr getShader() const { return m_shader; }
  const std::string &getName() const { return m_name; }

  void setPass(StringID pass, MaterialPassDefinition definition) {
    m_passes[pass] = std::move(definition);
  }

  std::optional<std::reference_wrapper<const MaterialPassDefinition>>
  getEntry(StringID pass) const {
    auto it = m_passes.find(pass);
    if (it != m_passes.end())
      return it->second;
    return std::nullopt;
  }

  StringID getRenderPassSignature(StringID pass) const {
    auto it = m_passes.find(pass);
    if (it == m_passes.end())
      return StringID{};
    return it->second.getRenderSignature();
  }

  void buildBindingCache() {
    m_bindingCache.clear();
    const auto addBindings = [&](const IShaderPtr &shader) {
      if (!shader)
        return;
      for (const auto &binding : shader->getReflectionBindings()) {
        m_bindingCache[StringID(binding.name)] = binding;
      }
    };

    addBindings(m_shader);
    for (const auto &[_, definition] : m_passes) {
      addBindings(definition.shaderSet.getShader());
    }
  }

  std::optional<std::reference_wrapper<const ShaderResourceBinding>>
  findBinding(StringID id) const {
    auto it = m_bindingCache.find(id);
    if (it != m_bindingCache.end())
      return it->second;
    return std::nullopt;
  }

  const std::unordered_map<StringID, MaterialPassDefinition, StringID::Hash> &
  getPasses() const {
    return m_passes;
  }

private:
  std::string m_name;
  IShaderPtr m_shader;
  std::unordered_map<StringID, MaterialPassDefinition, StringID::Hash> m_passes;
  std::unordered_map<StringID, ShaderResourceBinding> m_bindingCache;
};

} // namespace LX_core
