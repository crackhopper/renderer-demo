#pragma once

/**
 * Pipeline 基类
 *
 * 从这个基类，我们会构造若干个通用的pipeline子类。
 *
 */

#include "core/resources/vertex_buffer.hpp"
#include "vkp_pipeline_slot.hpp"
#include <memory>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace LX_core::graphic_backend {

class VulkanDevice;

struct PushConstantDetails {
  VkShaderStageFlags stageFlags;
  uint32_t offset;
  uint32_t size;
};

class VulkanPipelineBase;
using VulkanPipelinePtr = std::unique_ptr<VulkanPipelineBase>;
class VulkanPipelineBase {
protected:
  struct Token {};

public:
  explicit VulkanPipelineBase(Token, VulkanDevice &device, VkExtent2D extent,
                              const std::string &shaderName_,
                              PipelineSlotDetails *slots_, uint32_t slotCount_,
                              const PushConstantDetails &pushConstants_);
  ~VulkanPipelineBase();

  static VulkanPipelinePtr create(VulkanDevice &device, VkExtent2D extent,
                                  const std::string &shaderName_,
                                  PipelineSlotDetails *slots,
                                  uint32_t slotCount) {
    auto p = std::make_unique<VulkanPipelineBase>(
        Token{}, device, extent, shaderName_, slots, slotCount);
    p->loadShaders();
    p->createLayout();
    return p;
  }

  virtual std::string getPipelineId() const = 0;
  virtual std::string getShaderName() const = 0;
  virtual VertexFormat getVertexFormat() const = 0;

  virtual void loadShaders();
  virtual void createLayout();

  virtual VkPipelineVertexInputStateCreateInfo getVertexInputStateCreateInfo();
  virtual VkPipelineInputAssemblyStateCreateInfo
  getInputAssemblyStateCreateInfo();
  virtual VkPipelineShaderStageCreateInfo getVertexShaderStageCreateInfo();
  virtual VkPipelineShaderStageCreateInfo getFragmentShaderStageCreateInfo();
  virtual VkPipelineViewportStateCreateInfo getViewportStateCreateInfo();
  virtual VkPipelineDynamicStateCreateInfo getDynamicStateCreateInfo();
  virtual VkPipelineRasterizationStateCreateInfo getRasterizerStateCreateInfo();
  virtual VkPipelineMultisampleStateCreateInfo getMultisampleStateCreateInfo();
  virtual VkPipelineDepthStencilStateCreateInfo
  getDepthStencilStateCreateInfo();
  virtual VkPipelineColorBlendStateCreateInfo getColorBlendStateCreateInfo();

  VkPipeline buildGraphicsPpl(VkRenderPass renderPass);

  VkPipeline getPipelineHandle() const { return hPipeline; }

protected:
  VulkanDevice &device;
  VkDevice hDevice = VK_NULL_HANDLE;
  VkPipeline hPipeline = VK_NULL_HANDLE;
  VkPipelineLayout hLayout = VK_NULL_HANDLE;

  std::string shaderName;
  std::vector<PipelineSlotDetails> slots;
  PushConstantDetails pushConstants;

  VkShaderModule hVertShader = VK_NULL_HANDLE;
  VkShaderModule hFragShader = VK_NULL_HANDLE;

  // 输出区域
  VkOffset2D offset = {0, 0};
  VkExtent2D extent = {0, 0};

  // 多重采样
  VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;

  std::vector<VkVertexInputBindingDescription> m_viBindingDescriptions;
  std::vector<VkVertexInputAttributeDescription> m_viAttrDescriptions;
};

} // namespace LX_core::graphic_backend
