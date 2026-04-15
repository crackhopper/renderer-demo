#include "material.hpp"
#include "core/frame_graph/pass.hpp"

#include <algorithm>
#include <cstring>
#include <utility>

namespace LX_core {

namespace {

const RenderPassEntry *firstEntry(const MaterialTemplate &tmpl) {
  // Transitional: old accessors still assume a single Forward-only pass.
  // REQ-007 will route these through the render-signature pipeline later.
  auto opt = tmpl.getEntry(StringID("Forward"));
  if (opt)
    return &opt->get();
  return nullptr;
}

const ShaderResourceBinding *findMaterialUboBinding(const IShaderPtr &shader) {
  if (!shader)
    return nullptr;
  const auto &bindings = shader->getReflectionBindings();
  for (const auto &binding : bindings) {
    if (binding.type == ShaderPropertyType::UniformBuffer &&
        binding.name == "MaterialUBO") {
      return &binding;
    }
  }
  return nullptr;
}

} // namespace

/*****************************************************************
 * MaterialInstance
 *****************************************************************/

MaterialInstance::MaterialInstance(Token, MaterialTemplate::Ptr tmpl,
                                   ResourcePassFlag passFlag)
    : m_template(std::move(tmpl)), m_passFlag(passFlag) {
  if (!m_template) {
    return;
  }

  for (const auto &[pass, _] : m_template->getPasses()) {
    const auto flag = passFlagFromStringID(pass);
    if ((static_cast<uint32_t>(passFlag) & static_cast<uint32_t>(flag)) != 0) {
      m_enabledPasses.insert(pass);
    }
  }

  // Convention: the per-material UBO block is named `MaterialUBO` in GLSL.
  // Build the instance-side CPU buffer from the enabled pass shader set rather
  // than assuming the template shader is the one every pass uses.
  const ShaderResourceBinding *selectedBinding = nullptr;
  for (const auto &pass : getEnabledPasses()) {
    const auto *candidate = findMaterialUboBinding(getShaderInfo(pass));
    if (!candidate)
      continue;
    if (!selectedBinding) {
      selectedBinding = candidate;
      continue;
    }
    assert(selectedBinding->size == candidate->size &&
           selectedBinding->members == candidate->members &&
           "MaterialInstance enabled passes must agree on MaterialUBO layout");
  }
  if (!selectedBinding) {
    selectedBinding = findMaterialUboBinding(m_template->getShader());
  }
  if (selectedBinding) {
    m_uboBinding = selectedBinding;
    m_uboBuffer.assign(selectedBinding->size, uint8_t{0});
    m_uboResource = std::make_shared<UboByteBufferResource>(
        m_uboBuffer, selectedBinding->size, m_passFlag);
  }
}

void MaterialInstance::writeUboMember(StringID id, const void *src,
                                      size_t nbytes,
                                      ShaderPropertyType expected) {
  if (!m_uboBinding) {
    assert(false && "MaterialInstance has no UBO binding");
    return;
  }
  for (const auto &m : m_uboBinding->members) {
    if (StringID(m.name) != id)
      continue;
    assert(m.type == expected &&
           "MaterialInstance setter type does not match reflected member type");
    assert(static_cast<size_t>(m.offset) + nbytes <= m_uboBuffer.size() &&
           "UBO write would overflow the reflected buffer");
    std::memcpy(m_uboBuffer.data() + m.offset, src, nbytes);
    m_uboDirty = true;
    return;
  }
  assert(false && "MaterialInstance setter: member not found in UBO");
}

void MaterialInstance::setVec4(StringID id, const Vec4f &value) {
  writeUboMember(id, &value, sizeof(Vec4f), ShaderPropertyType::Vec4);
}

void MaterialInstance::setVec3(StringID id, const Vec3f &value) {
  // std140: vec3 occupies 12 bytes; if the next member packs into the
  // trailing 4 bytes of the 16-byte bucket, writing 16 would clobber it.
  writeUboMember(id, &value, sizeof(float) * 3, ShaderPropertyType::Vec3);
}

void MaterialInstance::setFloat(StringID id, float value) {
  writeUboMember(id, &value, sizeof(float), ShaderPropertyType::Float);
}

void MaterialInstance::setInt(StringID id, int32_t value) {
  writeUboMember(id, &value, sizeof(int32_t), ShaderPropertyType::Int);
}

void MaterialInstance::setTexture(StringID id, CombinedTextureSamplerPtr tex) {
  auto bindingOpt = m_template->findBinding(id);
  assert(bindingOpt && "texture binding not found in reflection");
  const auto type = bindingOpt->get().type;
  assert((type == ShaderPropertyType::Texture2D ||
          type == ShaderPropertyType::TextureCube) &&
         "setTexture target is not a sampled image binding");
  (void)type;
  m_textures[id] = std::move(tex);
}

void MaterialInstance::updateUBO() {
  if (!m_uboDirty || !m_uboResource)
    return;
  m_uboResource->setDirty();
  m_uboDirty = false;
}

std::vector<IRenderResourcePtr>
MaterialInstance::getDescriptorResources() const {
  std::vector<IRenderResourcePtr> out;

  if (m_uboResource && !m_uboBuffer.empty()) {
    out.push_back(m_uboResource);
  }

  // Collect textures sorted by (set << 16 | binding) so the descriptor
  // list appears in a deterministic order matching the shader layout.
  // `CombinedTextureSampler` inherits from `IRenderResource`, so the
  // conversion is a plain base-class upcast — no dynamic_cast needed.
  // Also stamps the reflected binding name onto each texture so backend
  // descriptor routing can find it by name.
  std::vector<std::pair<uint32_t, IRenderResourcePtr>> sorted;
  sorted.reserve(m_textures.size());
  for (const auto &[id, tex] : m_textures) {
    if (!tex)
      continue;
    auto b = m_template->findBinding(id);
    if (!b)
      continue;
    tex->setBindingName(id);
    const uint32_t key = (b->get().set << 16) | b->get().binding;
    sorted.emplace_back(key, std::static_pointer_cast<IRenderResource>(tex));
  }
  std::sort(sorted.begin(), sorted.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  for (auto &[_, r] : sorted) {
    out.push_back(std::move(r));
  }

  return out;
}

IShaderPtr MaterialInstance::getShaderInfo() const {
  return m_template ? m_template->getShader() : nullptr;
}

IShaderPtr MaterialInstance::getShaderInfo(StringID pass) const {
  if (!m_template)
    return nullptr;
  auto entry = m_template->getEntry(pass);
  if (entry) {
    auto shader = entry->get().shaderSet.getShader();
    if (shader)
      return shader;
  }
  return m_template->getShader();
}

RenderState MaterialInstance::getRenderState() const {
  const auto *entry = m_template ? firstEntry(*m_template) : nullptr;
  return entry ? entry->renderState : RenderState{};
}

StringID MaterialInstance::getRenderSignature(StringID pass) const {
  if (!m_template)
    return StringID{};
  StringID passSig = m_template->getRenderPassSignature(pass);
  StringID fields[] = {passSig};
  return GlobalStringTable::get().compose(TypeTag::MaterialRender, fields);
}

bool MaterialInstance::isPassEnabled(StringID pass) const {
  return m_enabledPasses.find(pass) != m_enabledPasses.end();
}

void MaterialInstance::setPassEnabled(StringID pass, bool enabled) {
  const bool currentlyEnabled = isPassEnabled(pass);
  if (enabled == currentlyEnabled)
    return;

  if (enabled) {
    m_enabledPasses.insert(pass);
    m_passFlag = m_passFlag | passFlagFromStringID(pass);
  } else {
    m_enabledPasses.erase(pass);
    uint32_t mask = static_cast<uint32_t>(m_passFlag);
    mask &= ~static_cast<uint32_t>(passFlagFromStringID(pass));
    m_passFlag = static_cast<ResourcePassFlag>(mask);
  }

  for (const auto &[_, callback] : m_passStateListeners) {
    if (callback)
      callback();
  }
}

std::vector<StringID> MaterialInstance::getEnabledPasses() const {
  std::vector<StringID> out;
  out.reserve(m_enabledPasses.size());
  for (const auto &pass : m_enabledPasses)
    out.push_back(pass);
  return out;
}

uint64_t MaterialInstance::addPassStateListener(std::function<void()> callback) {
  const uint64_t id = m_nextListenerId++;
  m_passStateListeners.emplace(id, std::move(callback));
  return id;
}

void MaterialInstance::removePassStateListener(uint64_t listenerId) {
  m_passStateListeners.erase(listenerId);
}

} // namespace LX_core
