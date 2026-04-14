# Renderer Demo — 项目速览

> 一个基于 Vulkan 的 C++20 渲染器示例工程，用来演练完整的材质系统 / pipeline identity / scene graph / frame graph。本目录的文档由 `/update-notes` 自动生成增量维护，最后同步：commit `0fe34a8`（2026-04-14）。

## 这是什么

renderer-demo 是一个教学型 / 实验型的 3D 渲染器。它不是一个游戏引擎，目标是把"材质如何驱动 pipeline"、"shader 反射如何替代硬编码绑定表"、"scene graph 如何产出 render item"这些课题的**干净实现**放在一起，方便对比和复用。

- **语言**: C++20
- **图形 API**: Vulkan（通过项目自写的薄层，不依赖 bgfx / Diligent 之类）
- **Shader**: GLSL → SPIR-V（`shaderc`），反射走 `SPIRV-Cross`
- **Window**: SDL3（默认）或 GLFW（可切）
- **构建**: CMake + Ninja / Make
- **平台**: Linux + Windows（cross-platform，PowerShell 脚本见 `scripts/`）

## 目录结构

- `src/core/` — 接口与纯数据（不依赖 Vulkan，只用 C++ 标准库）
- `src/infra/` — core 接口的具体实现（shader 编译、窗口、资源加载、GUI）
- `src/backend/vulkan/` — Vulkan 渲染后端
- `src/test/integration/` — 按模块一个可执行的集成测试
- `shaders/glsl/` — GLSL shader 源
- `openspec/specs/` — 每个能力的权威规范（阅读前必读）
- `openspec/changes/` — 在途变更 / 历史归档
- `docs/design/` — 深度设计文档（Chinese）
- `docs/requirements/` — 需求文档（draft + finished）
- `notes/` — **本目录**：面向新人的摘要与导航
- `.claude/commands/` — Claude Code slash 命令定义
- `.claude/agents/` — Claude Code 子代理定义

## 如何构建

Linux:

```bash
mkdir -p ./build && cd ./build
cmake .. -G Ninja                 # 或 Unix Makefiles
ninja test_shader_compiler        # 非 GPU 测试
ninja BuildTest                   # 全部集成测试
ninja Renderer                    # 主可执行（需 Vulkan device）
```

Windows：使用 `scripts/build-project.ps1`，它会自动处理 `SHADERC_DIR` / `SPIRV_CROSS_DIR`。

## 核心概念地图

| 概念 | 一句话 | 深入阅读 |
|------|--------|---------|
| **StringID / interning** | 所有字符串键（binding 名、pass 名、shader 名）都被 intern 成 `uint32_t`，比较与哈希成本 O(1) | `notes/subsystems/string-interning.md` |
| **RenderingItem** | 一次 draw call 的完整上下文：vertex/index buffer + shader + 资源列表 + PipelineKey + pass | `notes/subsystems/scene.md` |
| **PipelineKey** | 结构化 `StringID`，通过 `GlobalStringTable::compose` 由 object signature + material signature 组成 | `notes/subsystems/pipeline-identity.md` |
| **PipelineBuildInfo** | 脱离 backend 的 pipeline 构造输入包（shader bytecode + 反射 binding + layout + render state） | `notes/subsystems/pipeline-identity.md` |
| **MaterialInstance** | 唯一的 `IMaterial` 实现；基于 shader 反射自动管理 UBO 字节 buffer | `notes/subsystems/material-system.md` |
| **IShader / 反射** | `ShaderCompiler` + `ShaderReflector` + `ShaderImpl`，运行期从 GLSL 编出 SPIR-V 再解出 binding layout | `notes/subsystems/shader-system.md` |
| **Mesh / VertexLayout** | 顶点布局是 pipeline 的身份一部分；`VertexFactory` 管理可复用的顶点类型 | `notes/subsystems/geometry.md` |
| **Skeleton** | 骨骼动画资源，和 `Mesh` / `Material` 平级，住在 `core/resources/` | `notes/subsystems/skeleton.md` |
| **FrameGraph** | 描述一帧的渲染结构，包含若干 `FramePass`；是 pipeline 预构建扫描的入口 | `notes/subsystems/frame-graph.md` |
| **FramePass** | 一个渲染 pass 的单元，持有 `StringID name` + `RenderTarget target` + `RenderQueue queue` | `notes/subsystems/frame-graph.md` |
| **RenderQueue** | 单个 pass 内的 `RenderingItem` 队列，支持按 `PipelineKey` 排序和 unique `PipelineBuildInfo` 收集 | `notes/subsystems/frame-graph.md` |
| **PipelineCache** | Vulkan backend 的 pipeline 存储与 preload 入口，与 `VulkanResourceManager` 解耦 | `notes/subsystems/pipeline-cache.md` |
| **Vulkan backend** | `VulkanDevice` / `VulkanResourceManager` / `VulkanPipeline` / `VulkanCommandBuffer`，对 core 接口的实现层 | `notes/subsystems/vulkan-backend.md` |

## 找文档

- **工作流指南**: `GUIDES.md`（开发流程、slash 命令、子代理）
- **代理规则**: `AGENTS.md`（项目概览、C++ 规则、spec 索引）
- **Claude 索引**: `CLAUDE.md`（Specs Index + Design Docs Index）
- **权威 spec**: `openspec/specs/<capability>/spec.md`
- **深度设计**: `docs/design/<Name>.md`（中文）
- **历史需求**: `docs/requirements/finished/*.md`（项目演化路径）
- **归档变更**: `openspec/changes/archive/YYYY-MM-DD-<name>/`（每次落地的实施记录）
- **这里（notes/）**: 摘要 + 导航，帮新人快速建立心智模型

## 归档变更索引

每次落地的实施记录在 `openspec/changes/archive/` 下。按时间倒序：

| 归档 | 主题 |
|------|------|
| `2026-04-14-frame-graph-drives-rendering` | `FrameGraph` 真正驱动 backend 渲染路径：`VulkanRenderer` 持 `m_frameGraph`，`RenderQueue::buildFromScene` 接管 `RenderingItem` 构造 + `Scene::getSceneLevelResources` 合并 + `IRenderable::supportsPass` 过滤 |
| `2026-04-13-interning-pipeline-identity` | Pipeline identity 基于 `getRenderSignature(pass) → StringID` 结构化 intern |
| `2026-04-13-unify-material-system` | `MaterialInstance` 作为唯一 `IMaterial` 实现，基于 shader 反射驱动 UBO |
| `2026-04-13-ubo-member-reflection` | SPIRV-Cross 反射扩展：抽取 UBO 成员（`StructMemberInfo`） |
| `2026-04-13-pipeline-prebuilding` | `PipelineBuildInfo` / `PipelineCache` / `FrameGraph` / `ImageFormat` + `RenderTarget` 基础设施 |
| `2026-04-13-extend-string-table-compose` | `GlobalStringTable` 结构化 `compose` / `decompose` / `TypeTag` |
| `2026-04-10-pipeline-key-rendering-item` | `PipelineKey` + `RenderingItem.pipelineKey` 字段 |
| `2026-04-10-migrate-skeleton-to-resources` | Skeleton 作为独立资源管理器迁到 `core/resources/` |

下一批计划中的工作见 `docs/requirements/*.md` 顶层（未归档）。
