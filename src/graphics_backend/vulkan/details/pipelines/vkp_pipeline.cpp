#include "vkp_pipeline.hpp"

#include "../vk_device.hpp"
#include "../descriptors/vkd_descriptor_manager.hpp"
#include <fstream>

namespace LX_core::graphic_backend {

// 辅助函数：读取二进制文件
static std::vector<char> readFile(const std::string &filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("failed to open file: " + filename);
  size_t fileSize = (size_t)file.tellg();
  std::vector<char> buffer(fileSize);
  file.seekg(0);
  file.read(buffer.data(), fileSize);
  file.close();
  return buffer;
}

VulkanPipelineBase::VulkanPipelineBase(
    Token token, VulkanDevice &device, VkExtent2D extent,
    const std::string &shaderName_, PipelineSlotDetails *slots_,
    uint32_t slotCount_, const PushConstantDetails &pushConstants_)
    : device(device), hDevice(device.getHandle()), extent(extent),
      shaderName(shaderName_), slots(slots_, slots_ + slotCount_),
      pushConstants(pushConstants_) {}

VulkanPipelineBase::~VulkanPipelineBase() {
  if (hVertShader)
    vkDestroyShaderModule(hDevice, hVertShader, nullptr);
  hVertShader = VK_NULL_HANDLE;

  if (hFragShader)
    vkDestroyShaderModule(hDevice, hFragShader, nullptr);
  hFragShader = VK_NULL_HANDLE;

  if (hLayout)
    vkDestroyPipelineLayout(hDevice, hLayout, nullptr);
  hLayout = VK_NULL_HANDLE;

  if (hPipeline)
    vkDestroyPipeline(hDevice, hPipeline, nullptr);
  hPipeline = VK_NULL_HANDLE;
}

VkPipelineInputAssemblyStateCreateInfo
VulkanPipelineBase::getInputAssemblyStateCreateInfo() {
  VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
  inputAssembly.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  inputAssembly.primitiveRestartEnable = VK_FALSE; // 不常用。
  // 就代码会用这个 + TRIANGLE_STRIP
  // 。然后特殊的index来重启，切断之前的绘制。（从而节约drawcall） 现在 使用经过
  // Vertex Cache Optimization（如使用 Forsyth 或 Tipsy 算法处理后的
  // TRIANGLE_LIST）的性能，往往比单纯使用 TRIANGLE_STRIP 还要好。
  return inputAssembly;
}

VkPipelineShaderStageCreateInfo
VulkanPipelineBase::getVertexShaderStageCreateInfo() {
  VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
  vertShaderStageInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vertShaderStageInfo.module = hVertShader;
  vertShaderStageInfo.pName = "main";
  return vertShaderStageInfo;
}
VkPipelineShaderStageCreateInfo
VulkanPipelineBase::getFragmentShaderStageCreateInfo() {
  VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
  fragShaderStageInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  fragShaderStageInfo.module = hFragShader;
  fragShaderStageInfo.pName = "main";
  return fragShaderStageInfo;
}

VkPipelineViewportStateCreateInfo
VulkanPipelineBase::getViewportStateCreateInfo() {
  VkViewport viewport{};
  viewport.x = offset.x;
  viewport.y = offset.y;
  viewport.width = (float)extent.width;
  viewport.height = (float)extent.height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkRect2D scissor{};
  scissor.offset = offset;
  scissor.extent = extent;

  VkPipelineViewportStateCreateInfo viewportState{};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;
  viewportState.pViewports = &viewport;
  viewportState.pScissors = &scissor;

  return viewportState;
}

VkPipelineDynamicStateCreateInfo
VulkanPipelineBase::getDynamicStateCreateInfo() {
  // 动态参数设置： Viewport+Scissor
  std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                               VK_DYNAMIC_STATE_SCISSOR};

  VkPipelineDynamicStateCreateInfo dynamicState{};
  dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
  dynamicState.pDynamicStates = dynamicStates.data();
  return dynamicState;
}

VkPipelineRasterizationStateCreateInfo
VulkanPipelineBase::getRasterizerStateCreateInfo() {
  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode =
      VK_POLYGON_MODE_FILL; // Using any mode other than fill requires enabling
                            // a GPU feature.
  // 如果用不是1.0f的值，需要开启 wide lines 功能
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

  rasterizer.depthBiasEnable = VK_FALSE;
  rasterizer.depthBiasConstantFactor = 0.0f; // Optional
  rasterizer.depthBiasClamp = 0.0f;          // Optional
  rasterizer.depthBiasSlopeFactor = 0.0f;    // Optional
  return rasterizer;
}

VkPipelineMultisampleStateCreateInfo
VulkanPipelineBase::getMultisampleStateCreateInfo() {
  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.rasterizationSamples = msaaSamples;
  // 如果多重采样，但是想降低 fs 执行频率，修改下面两行的参数。
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.minSampleShading = 1.0f;          // Optional
  multisampling.pSampleMask = nullptr;            // Optional
  multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
  multisampling.alphaToOneEnable = VK_FALSE;      // Optional
  return multisampling;
}

VkPipelineDepthStencilStateCreateInfo
VulkanPipelineBase::getDepthStencilStateCreateInfo() {
  VkPipelineDepthStencilStateCreateInfo depthStencil{};
  depthStencil.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  // 测试fragment是否通过
  depthStencil.depthTestEnable = VK_TRUE;
  // 测试通过后，是否写入depth
  depthStencil.depthWriteEnable = VK_TRUE;

  // 更小的深度值可以写入
  depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

  // 特殊能力，仅保留depth在某个区间的fragment 进行测试。
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.minDepthBounds = 0.0f; // Optional
  depthStencil.maxDepthBounds = 1.0f; // Optional

  return depthStencil;
}

VkPipelineColorBlendStateCreateInfo
VulkanPipelineBase::getColorBlendStateCreateInfo() {
  // Color blending
  VkPipelineColorBlendAttachmentState colorBlendAttachment{};
  colorBlendAttachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment.blendEnable = VK_FALSE;
  colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;  // Optional
  colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
  colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;             // Optional
  colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;  // Optional
  colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
  colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;             // Optional

  VkPipelineColorBlendStateCreateInfo colorBlending{};
  colorBlending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &colorBlendAttachment;
  // 这些常量是全局混合的时候使用，现在基本不用
  colorBlending.blendConstants[0] = 0.0f; // Optional
  colorBlending.blendConstants[1] = 0.0f; // Optional
  colorBlending.blendConstants[2] = 0.0f; // Optional
  colorBlending.blendConstants[3] = 0.0f; // Optional
  return colorBlending;
}

VkPipelineVertexInputStateCreateInfo
VulkanPipelineBase::getVertexInputStateCreateInfo() {
  VertexFormat format = getVertexFormat(); // 假设子类通过此函数返回格式

  // 清空旧数据
  m_viBindingDescriptions.clear();
  m_viAttrDescriptions.clear();

  if (format == VertexFormat::Custom) {
    // 交由子类自行填充 bindingDescriptions 和 attributeDescriptions
    return {};
  }

  // 1. 定义 Binding (Slot 0)
  VkVertexInputBindingDescription binding{};
  binding.binding = 0;
  binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
  // 根据格式确定 stride
  uint32_t stride = 0;

  // 2. 定义 Attributes
  auto addAttr = [&](uint32_t loc, VkFormat vkFormat, uint32_t offset) {
    VkVertexInputAttributeDescription attr{};
    attr.binding = 0;
    attr.location = loc;
    attr.format = vkFormat;
    attr.offset = offset;
    m_viAttrDescriptions.push_back(attr);
  };

  switch (format) {
  case VertexFormat::Pos:
    stride = sizeof(VertexPos);
    addAttr(0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexPos, pos));
    break;

  case VertexFormat::PosColor:
    stride = sizeof(VertexPosColor);
    addAttr(0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexPosColor, pos));
    addAttr(1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexPosColor, color));
    break;

  case VertexFormat::PosUV:
    stride = sizeof(VertexPosUV);
    addAttr(0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexPosUV, pos));
    addAttr(1, VK_FORMAT_R32G32_SFLOAT, offsetof(VertexPosUV, uv));
    break;

  case VertexFormat::NormalTangent:
    stride = sizeof(VertexNormalTangent);
    addAttr(0, VK_FORMAT_R32G32B32_SFLOAT,
            offsetof(VertexNormalTangent, normal));
    addAttr(1, VK_FORMAT_R32G32B32A32_SFLOAT,
            offsetof(VertexNormalTangent, tangent));
    break;

  case VertexFormat::BoneWeight:
    stride = sizeof(VertexBoneWeight);
    addAttr(0, VK_FORMAT_R32G32B32A32_SINT,
            offsetof(VertexBoneWeight, boneIds));
    addAttr(1, VK_FORMAT_R32G32B32A32_SFLOAT,
            offsetof(VertexBoneWeight, weights));
    break;

  case VertexFormat::PosNormalUvBone:
    stride = sizeof(VertexPosNormalUvBone);
    addAttr(0, VK_FORMAT_R32G32B32_SFLOAT,
            offsetof(VertexPosNormalUvBone, pos));
    addAttr(1, VK_FORMAT_R32G32B32_SFLOAT,
            offsetof(VertexPosNormalUvBone, normal));
    addAttr(2, VK_FORMAT_R32G32_SFLOAT, offsetof(VertexPosNormalUvBone, uv));
    addAttr(3, VK_FORMAT_R32G32B32A32_SFLOAT,
            offsetof(VertexPosNormalUvBone, tangent));
    addAttr(4, VK_FORMAT_R32G32B32A32_SINT,
            offsetof(VertexPosNormalUvBone, boneIDs));
    addAttr(5, VK_FORMAT_R32G32B32A32_SFLOAT,
            offsetof(VertexPosNormalUvBone, boneWeights));
    break;
  }

  binding.stride = stride;
  m_viBindingDescriptions.push_back(binding);

  // 3. 装填 CreateInfo
  VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
  vertexInputInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputInfo.vertexBindingDescriptionCount =
      static_cast<uint32_t>(m_viBindingDescriptions.size());
  vertexInputInfo.pVertexBindingDescriptions = m_viBindingDescriptions.data();
  vertexInputInfo.vertexAttributeDescriptionCount =
      static_cast<uint32_t>(m_viAttrDescriptions.size());
  vertexInputInfo.pVertexAttributeDescriptions = m_viAttrDescriptions.data();

  return vertexInputInfo;
}

VkPipeline VulkanPipelineBase::buildGraphicsPpl(VkRenderPass renderPass) {
  // shader信息
  VkPipelineShaderStageCreateInfo stages[2]{};
  stages[0] = getVertexShaderStageCreateInfo();
  stages[1] = getFragmentShaderStageCreateInfo();

  // 顶点格式
  VkPipelineVertexInputStateCreateInfo vertexInputInfo =
      getVertexInputStateCreateInfo();

  // 输入装配
  VkPipelineInputAssemblyStateCreateInfo inputAssembly =
      getInputAssemblyStateCreateInfo();

  // viewport and scissor
  VkPipelineViewportStateCreateInfo viewportState =
      getViewportStateCreateInfo();

  // 动态参数
  VkPipelineDynamicStateCreateInfo dynamicState = getDynamicStateCreateInfo();

  // rasterizer配置
  VkPipelineRasterizationStateCreateInfo rasterizer =
      getRasterizerStateCreateInfo();

  // multisampling设置
  VkPipelineMultisampleStateCreateInfo multisampling =
      getMultisampleStateCreateInfo();

  // depth and stencil testing
  VkPipelineDepthStencilStateCreateInfo depthStencil =
      getDepthStencilStateCreateInfo();

  // 颜色混合
  VkPipelineColorBlendStateCreateInfo colorBlending =
      getColorBlendStateCreateInfo();

  // 创建 graphics pipeline
  VkGraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = stages;
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pDynamicState = &dynamicState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pDepthStencilState = &depthStencil; // Optional
  pipelineInfo.pColorBlendState = &colorBlending;

  pipelineInfo.layout = hLayout;

  // 绑定 render pass （用来指定渲染目标和附件）
  // 需要 renderpass 兼容当前的pipeline
  pipelineInfo.renderPass = renderPass;
  pipelineInfo.subpass = 0;
  // 绑定 base pipeline （可选，用来继承之前的 pipeline 配置）
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
  pipelineInfo.basePipelineIndex = -1;              // Optional

  if (vkCreateGraphicsPipelines(hDevice, VK_NULL_HANDLE, 1, &pipelineInfo,
                                nullptr, &hPipeline) != VK_SUCCESS) {
    throw std::runtime_error("failed to create graphics pipeline!");
  }
  return hPipeline;
}

void VulkanPipelineBase::loadShaders() {
  // 假设你的 Shader 命名规则是 shaderName.vert.spv 和 shaderName.frag.spv
  auto vertCode = readFile("shaders/bin/" + shaderName + ".vert.spv");
  auto fragCode = readFile("shaders/bin/" + shaderName + ".frag.spv");

  auto createModule = [this](const std::vector<char> &code) {
    VkShaderModuleCreateInfo createInfo{
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());
    VkShaderModule module;
    if (vkCreateShaderModule(hDevice, &createInfo, nullptr, &module) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create shader module!");
    }
    return module;
  };

  hVertShader = createModule(vertCode);
  hFragShader = createModule(fragCode);
}

void VulkanPipelineBase::createLayout() {
  // 1. 获取 DescriptorManager 实例
  // VulkanDevice 中持有 VulkanDevice 中持有管理器的引用
  auto &descriptorMgr = device.getDescriptorManager();

  // 2. 将 slots 按 setIndex 分组
  // 因为一个 PipelineLayout 可能包含多个 DescriptorSetLayout (Set 0, Set 1...)
  std::unordered_map<uint32_t, std::vector<PipelineSlotDetails>> setGroups;
  for (const auto &slot : slots) {
    setGroups[slot.setIndex].push_back(slot);
  }

  // 3. 为每个 SetIndex 获取对应的 VkDescriptorSetLayout
  std::vector<VkDescriptorSetLayout> setLayouts;
  // 排序确保 Set 的顺序是 0, 1, 2...
  for (uint32_t i = 0; i < setGroups.size(); ++i) {
    setLayouts.push_back(descriptorMgr.getOrCreateLayout(setGroups[i]));
  }

  // 4. 处理 Push Constant

  // 5. 创建 Pipeline Layout
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
  pipelineLayoutInfo.pSetLayouts = setLayouts.data();
  if (pushConstants.size > 0) {
    VkPushConstantRange range{};
    range.stageFlags = pushConstants.stageFlags;
    range.offset = pushConstants.offset;
    range.size = pushConstants.size;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &range;
  } else {
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;
  }

  if (vkCreatePipelineLayout(hDevice, &pipelineLayoutInfo, nullptr, &hLayout) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create pipeline layout!");
  }
}

} // namespace LX_core::graphic_backend
