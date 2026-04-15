#pragma once
#include "core/asset/shader.hpp"
#include <unordered_map>

namespace LX_infra {

class CompiledShader : public LX_core::IShader {
public:
  CompiledShader(std::vector<LX_core::ShaderStageCode> stages,
                 std::vector<LX_core::ShaderResourceBinding> bindings,
                 std::vector<LX_core::VertexInputAttribute> vertexInputs = {},
                 std::string logicalName = {});

  // --- IShader interface ---
  const std::vector<LX_core::ShaderStageCode> &getAllStages() const override;

  const std::vector<LX_core::ShaderResourceBinding> &
  getReflectionBindings() const override;

  const std::vector<LX_core::VertexInputAttribute> &
  getVertexInputs() const override;

  std::optional<std::reference_wrapper<const LX_core::ShaderResourceBinding>>
  findBinding(uint32_t set, uint32_t binding) const override;

  std::optional<std::reference_wrapper<const LX_core::ShaderResourceBinding>>
  findBinding(const std::string &name) const override;

  std::optional<std::reference_wrapper<const LX_core::VertexInputAttribute>>
  findVertexInput(uint32_t location) const override;

  size_t getProgramHash() const override;

  std::string getShaderName() const override { return m_logicalName; }

  // --- IRenderResource interface ---
  LX_core::ResourcePassFlag getPassFlag() const override {
    return LX_core::ResourcePassFlag::Forward;
  }
  const void *getRawData() const override { return nullptr; }
  u32 getByteSize() const override { return 0; }

private:
  void buildIndices();
  void computeHash();

  std::vector<LX_core::ShaderStageCode> m_stages;
  std::vector<LX_core::ShaderResourceBinding> m_bindings;
  std::vector<LX_core::VertexInputAttribute> m_vertexInputs;

  // Fast lookup: (set << 16 | binding) -> index in m_bindings
  std::unordered_map<uint32_t, size_t> m_setBindingIndex;
  // Fast lookup: name -> index in m_bindings
  std::unordered_map<std::string, size_t> m_nameIndex;
  // Fast lookup: location -> index in m_vertexInputs
  std::unordered_map<uint32_t, size_t> m_vertexInputIndex;

  size_t m_hash = 0;
  std::string m_logicalName;
};

} // namespace LX_infra
