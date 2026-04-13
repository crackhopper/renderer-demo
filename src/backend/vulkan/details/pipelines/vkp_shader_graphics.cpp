#include "vkp_shader_graphics.hpp"
#include "../vk_device.hpp"

namespace LX_core::backend {

VulkanShaderGraphicsPipeline::VulkanShaderGraphicsPipeline(
    Token t, VulkanDevice &device, const PipelineBuildInfo &buildInfo,
    std::string shaderName)
    : VulkanPipeline(t, device, buildInfo),
      m_shaderName(std::move(shaderName)) {}

VulkanPipelinePtr VulkanShaderGraphicsPipeline::create(
    VulkanDevice &device, const PipelineBuildInfo &buildInfo,
    VkRenderPass renderPass, std::string shaderName) {
  auto pipeline = VulkanPipelinePtr(new VulkanShaderGraphicsPipeline(
      Token{}, device, buildInfo, std::move(shaderName)));
  pipeline->loadShaders();
  pipeline->createLayout();
  pipeline->buildGraphicsPpl(renderPass);
  return pipeline;
}

} // namespace LX_core::backend
