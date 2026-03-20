#include "vk_resource_manager.hpp"
#include "vk_device.hpp"
#include <stdexcept>

namespace LX_core::graphic_backend {

VulkanResourceManager::VulkanResourceManager(VulkanDevice &device)
    : m_device(device) {}

void VulkanResourceManager::syncResource(const IRenderResourcePtr &cpuRes) {
  if (!cpuRes)
    return;

  void *handle = cpuRes->getResourceHandle();
  m_activeHandles.insert(handle);

  auto it = m_gpuResources.find(handle);
  if (it == m_gpuResources.end()) {
    m_gpuResources[handle] = createGpuResource(cpuRes);
    // 新创建的资源强制更新一次数据
    updateGpuResource(m_gpuResources[handle], cpuRes);
    cpuRes->clearDirty();
  } else if (cpuRes->isDirty()) {
    updateGpuResource(it->second, cpuRes);
    cpuRes->clearDirty();
  }
}

std::shared_ptr<VulkanAnyResource>
VulkanResourceManager::createGpuResource(const IRenderResourcePtr &cpuRes) {
  ResourceType type = cpuRes->getType();

  switch (type) {
  case ResourceType::VertexBuffer:
    return std::make_shared<VulkanAnyResource>(VulkanBuffer::create(
        m_device, cpuRes->getByteSize(),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));

  case ResourceType::IndexBuffer:
    return std::make_shared<VulkanAnyResource>(VulkanBuffer::create(
        m_device, cpuRes->getByteSize(),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));

  case ResourceType::UniformBuffer:
    return std::make_shared<VulkanAnyResource>(VulkanBuffer::create(
        m_device, cpuRes->getByteSize(), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));

  case ResourceType::VertexShader:
  case ResourceType::FragmentShader: {
    auto shaderRes = std::static_pointer_cast<VulkanShader>(cpuRes);
    // 假设你有一个读取文件转 vector<char> 的工具函数
    // std::vector<char> code = readFile(shaderRes->getShaderName());
    VkShaderStageFlagBits stage = (type == ResourceType::VertexShader)
                                      ? VK_SHADER_STAGE_VERTEX_BIT
                                      : VK_SHADER_STAGE_FRAGMENT_BIT;
    return std::make_shared<VulkanAnyResource>(
        VulkanShader::create(m_device, {}, stage));
  }

  default:
    throw std::runtime_error("Unsupported resource type for GPU creation");
  }
}

void VulkanResourceManager::updateGpuResource(
    std::shared_ptr<VulkanAnyResource> &gpuRes,
    const IRenderResourcePtr &cpuRes) {
  std::visit(
      [&](auto &&res) {
        using T = std::decay_t<decltype(res)>;
        if constexpr (std::is_same_v<T, VulkanBufferPtr>) {
          // 如果是 Host Visible (Uniform)，直接 map/memcpy
          // 如果是 Device Local (Vertex/Index)，初级架构建议直接
          // uploadData（内部处理 staging）
          res->uploadData(cpuRes->getRawData(), cpuRes->getByteSize());
        } else if constexpr (std::is_same_v<T, VulkanTexturePtr>) {
          // 处理纹理上传逻辑...
        }
      },
      *gpuRes);
}

void VulkanResourceManager::collectGarbage() {
  for (auto it = m_gpuResources.begin(); it != m_gpuResources.end();) {
    if (m_activeHandles.find(it->first) == m_activeHandles.end()) {
      it = m_gpuResources.erase(it);
    } else {
      ++it;
    }
  }
  m_activeHandles.clear();
}

// 辅助查找宏，简化代码
#define GET_RESOURCE_IMPL(ReturnType, VariantType)                             \
  auto it = m_gpuResources.find(handle);                                       \
  if (it != m_gpuResources.end()) {                                            \
    if (auto resPtr = std::get_if<VariantType>(&(*(it->second)))) {            \
      return resPtr->get();                                                    \
    }                                                                          \
  }                                                                            \
  return nullptr;

VulkanBuffer *VulkanResourceManager::getBuffer(void *handle) {
  GET_RESOURCE_IMPL(VulkanBuffer *, VulkanBufferPtr);
}

VulkanTexture *VulkanResourceManager::getTexture(void *handle) {
  GET_RESOURCE_IMPL(VulkanTexture *, VulkanTexturePtr);
}

VulkanShader *VulkanResourceManager::getShader(void *handle) {
  GET_RESOURCE_IMPL(VulkanShader *, VulkanShaderPtr);
}

} // namespace LX_core::graphic_backend