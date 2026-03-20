#pragma once
#include "core/resources/vertex_buffer.hpp"
#include "vkp_pipeline.hpp"
#include "vkp_pipeline_slot.hpp"
#include <vector>
#include <vulkan/vulkan.h>

#include "core/scene/camera.hpp"
#include "core/scene/components/material.hpp"
#include "core/scene/components/skeleton.hpp"
#include "core/scene/light.hpp"
#include "core/scene/object.hpp"

namespace LX_core::graphic_backend {

// 定义与 Shader 匹配的 Push Constant 结构 (C++ 侧)
using BlinnPhongPushConstant = PC_BlinnPhong;

// descriptor 参数
// 注意： 只有buffer需要描述size， sampler 不需要
// TODO: 未来从 shader 反射获取
constexpr PipelineSlotDetails VkPipelineBlinnPhongSlotDetails[] = {
    {PipelineSlotId::LightUBO, ResourceType::UniformBuffer,
     PipelineSlotStage::ALL, 0, 0, DirectionalLightUBO::ResourceSize},
    {PipelineSlotId::CameraUBO, ResourceType::UniformBuffer,
     PipelineSlotStage::ALL, 1, 0, CameraUBO::ResourceSize},
    {PipelineSlotId::MaterialUBO, ResourceType::UniformBuffer,
     PipelineSlotStage::FRAGMENT, 2, 0, MaterialBlinnPhongUBO::ResourceSize},
    {PipelineSlotId::AlbedoTexture, ResourceType::CombinedImageSampler,
     PipelineSlotStage::FRAGMENT, 2, 1, 0},
    {PipelineSlotId::NormalTexture, ResourceType::CombinedImageSampler,
     PipelineSlotStage::ALL, 2, 2, 0},
    {PipelineSlotId::SkeletonUBO, ResourceType::UniformBuffer,
     PipelineSlotStage::VERTEX, 3, 0, SkeletonUBO::ResourceSize},
};

class VkPipelineBinnPhong : public VulkanPipelineBase {
  using VertexType = VertexPosNormalUvBone;

public:
  VkPipelineBinnPhong(Token t, VulkanDevice &device, VkExtent2D extent,
                      const std::string &shaderName_,
                      PipelineSlotDetails *slots_, uint32_t slotCount_,
                      const PushConstantDetails &pushConstants_)
      : VulkanPipelineBase(t, device, extent, shaderName_, slots_, slotCount_,
                           pushConstants_) {}

  static VulkanPipelinePtr create(VulkanDevice &device, VkExtent2D extent) {
    // 1. 创建子类实例
    // 注意：这里 shaderName 和 slots 直接从当前文件的常量获取
    PushConstantDetails pushConstant;
    pushConstant.stageFlags =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstant.size = sizeof(BlinnPhongPushConstant);
    pushConstant.offset = 0;

    auto p = std::make_unique<VkPipelineBinnPhong>(
        Token{}, device, extent, "blinnphong_0",
        const_cast<PipelineSlotDetails *>(VkPipelineBlinnPhongSlotDetails),
        static_cast<uint32_t>(sizeof(VkPipelineBlinnPhongSlotDetails) /
                              sizeof(PipelineSlotDetails)),
        pushConstant);

    // 2. 执行基类的初始化流程
    p->loadShaders();
    p->createLayout();
    return p;
  }

  VkPipelineVertexInputStateCreateInfo getVertexInputStateCreateInfo() override;

  VertexFormat getVertexFormat() const override {
    return VertexFormat::PosNormalUvBone;
  }
  std::string getShaderName() const override { return shaderName; }
  std::string getPipelineId() const override { return pipelineId; }

private:
  std::string pipelineId = "blinnphong";
  std::string shaderName = "blinnphong_0";
  std::vector<VkDescriptorSetLayout> m_descriptorLayouts;
};
} // namespace LX_core::graphic_backend
