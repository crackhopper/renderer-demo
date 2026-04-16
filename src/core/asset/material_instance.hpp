#pragma once

#include "core/asset/material_template.hpp"
#include "core/asset/texture.hpp"
#include "core/math/vec.hpp"
#include "core/rhi/render_resource.hpp"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace LX_core {

class MaterialParameterDataResource : public IRenderResource {
public:
  MaterialParameterDataResource(std::vector<uint8_t> &buffer, uint32_t byteSize,
                                StringID bindingName,
                                ResourceType resType = ResourceType::UniformBuffer)
      : m_buffer(&buffer), m_byteSize(byteSize),
        m_bindingName(bindingName), m_resType(resType) {}
  ResourceType getType() const override { return m_resType; }
  const void *getRawData() const override { return m_buffer->data(); }
  u32 getByteSize() const override { return m_byteSize; }

  StringID getBindingName() const override { return m_bindingName; }

private:
  std::vector<uint8_t> *m_buffer;
  uint32_t m_byteSize;
  StringID m_bindingName;
  ResourceType m_resType;
};

struct MaterialBufferSlot {
  StringID bindingName;
  const ShaderResourceBinding *binding = nullptr;
  std::vector<uint8_t> buffer;
  IRenderResourcePtr resource;
  bool dirty = false;
};

struct PassMaterialOverride {
  std::unordered_map<StringID, MaterialBufferSlot, StringID::Hash> bufferSlots;
  std::unordered_map<StringID, CombinedTextureSamplerPtr, StringID::Hash>
      textures;
};

class MaterialInstance {
  struct Token {};

public:
  using Ptr = std::shared_ptr<MaterialInstance>;

  MaterialInstance(Token, MaterialTemplate::Ptr tmpl);

  static Ptr create(MaterialTemplate::Ptr tmpl) {
    return std::make_shared<MaterialInstance>(Token{}, std::move(tmpl));
  }

  MaterialInstance(const MaterialInstance &) = delete;
  MaterialInstance &operator=(const MaterialInstance &) = delete;
  MaterialInstance(MaterialInstance &&) = delete;
  MaterialInstance &operator=(MaterialInstance &&) = delete;

  std::vector<IRenderResourcePtr> getDescriptorResources(StringID pass) const;
  IShaderPtr getShaderInfo(StringID pass) const;
  RenderState getRenderState(StringID pass) const;
  StringID getRenderSignature(StringID pass) const;

  // Primary API: write buffer parameter by binding name + member name.
  void setParameter(StringID bindingName, StringID memberName, float value);
  void setParameter(StringID bindingName, StringID memberName, int32_t value);
  void setParameter(StringID bindingName, StringID memberName,
                    const Vec3f &value);
  void setParameter(StringID bindingName, StringID memberName,
                    const Vec4f &value);
  void setParameter(StringID pass, StringID bindingName, StringID memberName,
                    float value);
  void setParameter(StringID pass, StringID bindingName, StringID memberName,
                    int32_t value);
  void setParameter(StringID pass, StringID bindingName, StringID memberName,
                    const Vec3f &value);
  void setParameter(StringID pass, StringID bindingName, StringID memberName,
                    const Vec4f &value);

  // Legacy convenience setters: search across all buffer slots by member name.
  // Assert if ambiguous (multiple slots contain the same member name).
  void setVec4(StringID id, const Vec4f &value);
  void setVec3(StringID id, const Vec3f &value);
  void setFloat(StringID id, float value);
  void setInt(StringID id, int32_t value);

  void setTexture(StringID id, CombinedTextureSamplerPtr tex);
  void setTexture(StringID pass, StringID id, CombinedTextureSamplerPtr tex);

  void syncGpuData();

  MaterialTemplate::Ptr getTemplate() const { return m_template; }

  // Multi-buffer accessors.
  size_t getBufferSlotCount() const { return m_bufferSlots.size(); }
  const std::vector<uint8_t> &getParameterBuffer(StringID bindingName) const;
  const ShaderResourceBinding *getParameterBinding(StringID bindingName) const;
  const std::vector<uint8_t> &getParameterBuffer(StringID pass,
                                                 StringID bindingName) const;
  const ShaderResourceBinding *getParameterBinding(StringID pass,
                                                   StringID bindingName) const;
  // Single-slot shortcuts (assert if multiple slots exist).
  const std::vector<uint8_t> &getParameterBuffer() const;
  const ShaderResourceBinding *getParameterBinding() const;

  bool isPassEnabled(StringID pass) const;
  void setPassEnabled(StringID pass, bool enabled);
  std::vector<StringID> getEnabledPasses() const;
  uint64_t addPassStateListener(std::function<void()> callback);
  void removePassStateListener(uint64_t listenerId);

private:
  void writeSlotMember(MaterialBufferSlot &slot, StringID memberName,
                       const void *src, size_t nbytes,
                       ShaderPropertyType expected);
  MaterialBufferSlot *findSlotByMember(StringID memberName);
  MaterialBufferSlot *findSlot(StringID bindingName);
  const MaterialBufferSlot *findSlot(StringID bindingName) const;
  MaterialBufferSlot *findSlot(StringID pass, StringID bindingName);
  const MaterialBufferSlot *findSlot(StringID pass, StringID bindingName) const;
  MaterialBufferSlot &ensurePassOverrideSlot(StringID pass, StringID bindingName);
  bool hasDefinedPass(StringID pass) const;

  MaterialTemplate::Ptr m_template;
  std::vector<MaterialBufferSlot> m_bufferSlots;

  std::unordered_map<StringID, CombinedTextureSamplerPtr> m_textures;
  std::unordered_map<StringID, PassMaterialOverride, StringID::Hash>
      m_passOverrides;
  std::unordered_set<StringID, StringID::Hash> m_enabledPasses;
  std::unordered_map<uint64_t, std::function<void()>> m_passStateListeners;
  uint64_t m_nextListenerId = 1;
};

using MaterialInstancePtr = MaterialInstance::Ptr;

} // namespace LX_core
