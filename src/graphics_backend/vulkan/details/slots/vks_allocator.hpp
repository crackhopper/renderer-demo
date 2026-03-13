#pragma once
#include "../vk_device.hpp"
#include <array>
#include <cassert>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

namespace LX_core::graphic_backend {

enum DescriptorSetLayoutIndex {
  DSLI_Camera = 0,
  DSLI_Light = 1,
  DSLI_Material = 2,
  DSLI_Skeleton = 3,

  DSLI_NUM_LAYOUT = 4,
};

struct DescriptorSetLayoutCreateInfo {
  DescriptorSetLayoutIndex index;
  VkDescriptorSetLayout layout = VK_NULL_HANDLE;
  VkDescriptorSetLayoutCreateInfo layoutInfo;
  std::vector<VkDescriptorSetLayoutBinding> bindings;
  std::vector<VkDescriptorPoolSize> poolSizes;
};

// 描述符集分配器
// - 描述符集，需要从描述符池来分配。这里简单做法是，用一个足够大的描述符池。

template <typename Derived, typename Data> class ResourceBindingBase;

class VulkanDescriptorAllocator;
using VulkanDescriptorAllocatorPtr = std::unique_ptr<VulkanDescriptorAllocator>;
// 会保存在 VulkanDevice 中。通过 getDescriptorAllocator() 来获取。
class VulkanDescriptorAllocator {
  struct Token {};

public:
  VulkanDescriptorAllocator(Token, VulkanDevice &device) : device(device) {
    // 风险点 1: 硬编码的池大小。
    // 如果渲染对象极多（如成千上万个材质实例），vkAllocateDescriptorSets
    // 会失败。 改进建议：实现一个 Pool 链，当当前池满时自动创建新池。
    std::array<VkDescriptorPoolSize, 3> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = uniformCount;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = samplerCount;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[2].descriptorCount = storageCount;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = maxSetCount;
    // 注意：这里没有设置 VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT
    // 因为我们目前采用的是手动 freeSets 缓存复用逻辑，而不是真正归还给 Vulkan
    // 池。

    if (vkCreateDescriptorPool(device.getHandle(), &poolInfo, nullptr,
                               &hPool) != VK_SUCCESS) {
      throw std::runtime_error("failed to create descriptor pool!");
    }
  }
  ~VulkanDescriptorAllocator() {
    for (auto &[layout, sets] : usedSets) {
      assert(sets.empty());
    }
    for (auto &[name, info] : layoutInfo) {
      vkDestroyDescriptorSetLayout(device.getHandle(), info.layout, nullptr);
    }
  }

  template <typename T, typename Data>
  bool allocate(ResourceBindingBase<T, Data> &resourceBinding);
  template <typename T, typename Data>
  void free(ResourceBindingBase<T, Data> &resourceBinding);

  static VulkanDescriptorAllocatorPtr create(VulkanDevice &device) {
    return std::make_unique<VulkanDescriptorAllocator>(Token{}, device);
  }

private:
  VulkanDevice &device;
  VkDescriptorPool hPool = VK_NULL_HANDLE;

  std::unordered_map<DescriptorSetLayoutIndex, DescriptorSetLayoutCreateInfo>
      layoutInfo;
  // 已分配列表，方便复用
  std::unordered_map<VkDescriptorSetLayout, std::vector<VkDescriptorSet>>
      usedSets;
  std::unordered_map<VkDescriptorSetLayout, std::vector<VkDescriptorSet>>
      freeSets;

  uint32_t maxSetCount = 1000; // 默认最大 descriptor set 数量
  uint32_t samplerCount = 1000;
  uint32_t uniformCount = 1000;
  uint32_t storageCount = 1000;
};

template <typename T, typename Data>
bool VulkanDescriptorAllocator::allocate(
    ResourceBindingBase<T, Data> &resourceBinding) {
  assert(hPool != VK_NULL_HANDLE);
  auto layout = resourceBinding.getLayoutHandle();
  if (layout != VK_NULL_HANDLE && !freeSets[layout].empty()) {
    resourceBinding.descriptorSet = freeSets[layout].back();
    freeSets[layout].pop_back();
    usedSets[layout].push_back(resourceBinding.descriptorSet);
    return true;
  }
  if (layout == VK_NULL_HANDLE) {
    auto layoutCreateInfo = resourceBinding.getLayoutCreateInfo();
    if (layoutInfo.find(layoutCreateInfo.index) != layoutInfo.end()) {
      layout = layoutInfo[layoutCreateInfo.index].layout;
    } else {
      if (vkCreateDescriptorSetLayout(device.getHandle(),
                                      &layoutCreateInfo.layoutInfo, nullptr,
                                      &layout) != VK_SUCCESS) {
        return false;
      }
      layoutCreateInfo.layout = layout;
      layoutInfo[layoutCreateInfo.index] = layoutCreateInfo;
    }
    resourceBinding.layout = layout;
  }

  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = hPool;
  allocInfo.pSetLayouts = &layout;
  allocInfo.descriptorSetCount = 1;

  VkResult result = vkAllocateDescriptorSets(device.getHandle(), &allocInfo,
                                             &resourceBinding.descriptorSet);
  usedSets[layout].push_back(resourceBinding.descriptorSet);
  return result == VK_SUCCESS;
}

template <typename T, typename Data>
void VulkanDescriptorAllocator::free(
    ResourceBindingBase<T, Data> &resourceBinding) {
  auto layout = resourceBinding.getLayoutHandle();
  if (layout == VK_NULL_HANDLE) {
    return;
  }
  auto it = std::find(usedSets[layout].begin(), usedSets[layout].end(),
                      resourceBinding.descriptorSet);
  if (it != usedSets[layout].end()) {
    usedSets[layout].erase(it);
    freeSets[layout].push_back(resourceBinding.descriptorSet);
  }
}

// 描述符集绑定的基类。
template <typename Derived, typename Data> class ResourceBindingBase {
protected:
  struct Token {};

public:
  using Ptr = std::unique_ptr<Derived>;
  using Base = ResourceBindingBase<Derived, Data>;

  ResourceBindingBase(Token, VulkanDevice &device,
                      VulkanDescriptorAllocator &allocator);
  virtual ~ResourceBindingBase();

  static Ptr create(VulkanDevice &device, VulkanDescriptorAllocator &allocator,
                    const Data &data) {
    auto p = std::make_unique<Derived>(Token{}, device, allocator, data);
    p->init();
    return p;
  }
  virtual void init();

  virtual DescriptorSetLayoutCreateInfo getLayoutCreateInfo() = 0;

  // 更新描述符集。
  virtual void update(VulkanDevice &device, const Data &data) = 0;

  VkDescriptorSet getDescriptorSetHandle() const { return descriptorSet; }
  VkDescriptorSetLayout getLayoutHandle() const { return layout; }

protected:
  VkDevice hDevice;
  VulkanDescriptorAllocator &allocator;
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
  VkDescriptorSetLayout layout = VK_NULL_HANDLE;
  friend class VulkanDescriptorAllocator;
};

template <typename Derived, typename Data>
ResourceBindingBase<Derived, Data>::ResourceBindingBase(
    Token, VulkanDevice &device, VulkanDescriptorAllocator &allocator)
    : hDevice(device.getHandle()), allocator(allocator) {}
template <typename Derived, typename Data>
void ResourceBindingBase<Derived, Data>::init() {
  allocator.allocate(*this);
}

template <typename Derived, typename Data>
ResourceBindingBase<Derived, Data>::~ResourceBindingBase() {
  if (descriptorSet != VK_NULL_HANDLE) {
    allocator.free(*this);
    descriptorSet = VK_NULL_HANDLE;
    layout = VK_NULL_HANDLE;
  }
}

} // namespace LX_core::graphic_backend