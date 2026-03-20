#pragma once
#include <string>
#include <vector>

namespace LX_core::graphic_backend {

class VulkanDevice;
class VulkanShader;
using VulkanShaderPtr = std::unique_ptr<VulkanShader>;

class VulkanShader {
  struct Token {};

public:
  VulkanShader(Token, const VulkanDevice &_device, const std::string &name,
               VkShaderStageFlagBits stage);
  ~VulkanShader();

  static VulkanShaderPtr create(const VulkanDevice &_device,
                                const std::string &name,
                                VkShaderStageFlagBits stage) {
    return std::make_unique<VulkanShader>(Token{}, _device, name, stage);
  }

  VkPipelineShaderStageCreateInfo getStageCreateInfo() const;
  VkShaderStageFlagBits getStage() const { return stage; }
  VkShaderModule getHandle() const { return module; }

private:
  VkDevice device = VK_NULL_HANDLE;
  VkShaderModule module = VK_NULL_HANDLE;
  VkShaderStageFlagBits stage;
};

} // namespace LX_core::graphic_backend