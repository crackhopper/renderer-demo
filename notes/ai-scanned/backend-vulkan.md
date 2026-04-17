# Backend Vulkan 扫描：句柄契约和设备选择上的硬错误

这部分问题集中在 Vulkan 后端和窗口后端的交界处。它们的共同特点是，平时在单一环境里可能不容易暴露，但一旦切换平台、GPU 类型或窗口后端，就会直接演化成崩溃、无效句柄或初始化失败。

## 严重问题

| 严重度 | 位置 | 现象 | 影响 |
| --- | --- | --- | --- |
| 严重 | [src/infra/window/glfw_window.cpp](../../src/infra/window/glfw_window.cpp)、[src/backend/vulkan/details/device.cpp](../../src/backend/vulkan/details/device.cpp) | GLFW 路径把 `VkSurfaceKHR` 堆分配成 `new VkSurfaceKHR(...)`，设备层再把这个指针值直接当成 `VkSurfaceKHR` 使用 | GLFW 后端下 `m_surface` 不是 Vulkan surface，而是“指向 surface 的指针地址”；后续 `vkGetPhysicalDeviceSurfaceSupportKHR`、`vkDestroySurfaceKHR` 都会吃到错误句柄 |
| 高 | [src/backend/vulkan/details/device.cpp](../../src/backend/vulkan/details/device.cpp) | `isDeviceSuitable()` 强制要求 `VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU` | 注释写的是“优先独显，兜底第一块可用 GPU”，实现却会把只有集显的机器全部判成不可用 |
| 高 | [src/backend/vulkan/details/device.cpp](../../src/backend/vulkan/details/device.cpp) | `findQueueFamilies()` 直接写成员 `m_queueIndices`，没有用局部临时值 | 探测多块物理设备时，前一块卡留下的队列族状态可能污染后一块卡的判断结果 |

## 证据

### 1. GLFW Surface 句柄在窗口层和设备层之间已经失真

- [src/infra/window/glfw_window.cpp](../../src/infra/window/glfw_window.cpp) 第 86-90 行把 surface 包成 `new VkSurfaceKHR(...)`
- [src/backend/vulkan/details/device.cpp](../../src/backend/vulkan/details/device.cpp) 第 168-170 行把返回值直接强转回 `VkSurfaceKHR`
- [src/infra/window/glfw_window.cpp](../../src/infra/window/glfw_window.cpp) 第 94-97 行的 `destroyGraphicsHandle()` 还是空实现，连那块堆内存都没释放

这个问题不是风格争议，而是实打实的 ABI/句柄契约错误。SDL 路径返回的是“句柄值”，GLFW 路径返回的是“句柄地址”，同一个接口出现了两套语义。

### 2. 设备选择策略和注释相反

- [src/backend/vulkan/details/device.cpp](../../src/backend/vulkan/details/device.cpp) 第 353-367 行：`isDeviceSuitable()` 只有在独显时才返回 `true`
- 同文件第 381-395 行：`pickPhysicalDevice()` 的注释和分支结构明显想表达“先记住可用卡，再看看有没有更好的独显”

这会让没有独显的 Linux 笔记本、虚拟机、CI runner、开发容器全部在初始化阶段失败。

### 3. 队列族探测把“本次探测结果”和“设备对象全局状态”混在一起

- [src/backend/vulkan/details/device.cpp](../../src/backend/vulkan/details/device.cpp) 第 293-325 行：`findQueueFamilies()` 每次扫描都直接写 `m_queueIndices`
- 同文件第 359 行和第 403 行分别在“探测候选设备”和“确认最终设备”时重复调用这个函数

这类写法让函数不再是纯查询，而是带副作用的探测器。后续只要设备枚举顺序、surface 支持矩阵或调试流程变化，就很容易引入难复现的初始化错误。

## 设计债务

| 类型 | 位置 | 说明 |
| --- | --- | --- |
| 规范违背 | [src/backend/vulkan/vulkan_renderer.hpp](../../src/backend/vulkan/vulkan_renderer.hpp) 第 25 行 | `VulkanRenderer` 用 `gpu::Renderer* p_impl` 手工管理实现对象，和项目自己的“禁止裸指针持有对象引用/所有权”规范冲突 |
| 重复代码 | [src/backend/vulkan/vulkan_renderer.cpp](../../src/backend/vulkan/vulkan_renderer.cpp) 第 27-30 行、[src/backend/vulkan/details/commands/command_buffer.cpp](../../src/backend/vulkan/details/commands/command_buffer.cpp) 第 16-20 行 | `envEnabled()` 被复制了至少两份，后续调试开关行为很难统一 |
| 死代码 | [src/backend/vulkan/vulkan_renderer.cpp](../../src/backend/vulkan/vulkan_renderer.cpp) 第 32-64 行、第 347-357 行 | `vkResultToString()`、`debugLog()`、`chooseSwapSurfaceFormat()` 目前没有实际调用，说明后端里已有一段“准备做但没接上”的调试/封装层 |

## 建议的修正方向

| 优先级 | 建议 |
| --- | --- |
| P0 | 统一 `Window::createGraphicsHandle()` 的契约，所有后端都返回“句柄值”而不是“堆上的句柄副本” |
| P0 | 把 `findQueueFamilies()` 改成纯函数，使用局部 `QueueFamilyIndices` 返回值 |
| P0 | 设备选择先判断“是否可用”，再单独做“独显优先”排序，不要把“可用”和“偏好”写成同一个布尔条件 |
| P1 | 把 `VulkanRenderer` 的 `p_impl` 改为 `std::unique_ptr<VulkanRendererImpl>` |
| P1 | 合并环境变量读取逻辑，删掉未接线的 helper |

## 继续阅读

- [Infra 与构建层扫描](infra-and-build.md)
- [VulkanBackend](../subsystems/vulkan-backend.md)
- [Renderer Backend Vulkan Spec](../../openspec/specs/renderer-backend-vulkan/spec.md)
