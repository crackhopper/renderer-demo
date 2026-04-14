# Frame Graph

> 一帧的渲染结构：若干 `FramePass` 组成，每个 pass 有自己的 `RenderQueue`。`FrameGraph` 由 `VulkanRenderer::Impl` 持有（成员 `m_frameGraph`），生命周期与 scene 绑定。它既是 pipeline 预构建的扫描入口，也是 draw loop 的遍历入口——backend 在每帧 `draw()` 里直接 `for pass in getPasses() × item in pass.queue.getItems()`。
>
> 权威 spec: `openspec/specs/frame-graph/spec.md`

## 核心抽象

### `FramePass` (`src/core/scene/frame_graph.hpp:14`)

```cpp
struct FramePass {
    StringID    name;      // Pass_Forward / Pass_Shadow / Pass_Deferred
    RenderTarget target;   // color format + depth format + sample count
    RenderQueue  queue;
};
```

`name` 是 `StringID`，对齐 `RenderQueue::buildFromScene(scene, pass)` 和 `IRenderable::supportsPass(pass)` 的参数类型。

### `FrameGraph` (`src/core/scene/frame_graph.hpp:22`)

```cpp
class FrameGraph {
public:
    void addPass(FramePass pass);
    void buildFromScene(const Scene &scene);
    std::vector<PipelineBuildInfo> collectAllPipelineBuildInfos() const;
    const std::vector<FramePass> &getPasses() const;
    std::vector<FramePass> &getPasses();
};
```

- `buildFromScene`: 委托给每个 pass 的 `queue.buildFromScene(scene, pass.name)`。FrameGraph 本身不构造 `RenderingItem`。
- `collectAllPipelineBuildInfos`: 汇总所有 queue 的 unique `PipelineBuildInfo`，按 `PipelineKey` 全局去重

### `RenderQueue` (`src/core/scene/render_queue.hpp:12`)

```cpp
class RenderQueue {
public:
    void addItem(RenderingItem item);
    void clearItems();
    void sort();                                 // 按 PipelineKey 稳定排序
    const std::vector<RenderingItem> &getItems() const;
    std::vector<PipelineBuildInfo> collectUniquePipelineBuildInfos() const;

    /// 从 scene 构建 queue：clearItems → 过滤 supportsPass → 构造 item
    /// → 合并 scene.getSceneLevelResources() → sort。
    void buildFromScene(const Scene &scene, StringID pass);
};
```

`RenderQueue::buildFromScene` 是 **唯一** 的 `RenderingItem` 构造入口。`Scene` 不再提供 item factory 方法。

### `ImageFormat` + `RenderTarget` (`src/core/gpu/`)

- `ImageFormat` (`image_format.hpp`) — `uint8_t` 枚举：`RGBA8` / `BGRA8` / `R8` / `D32Float` / `D24UnormS8` / `D32FloatS8`
- `RenderTarget` (`render_target.hpp:11`) — `{colorFormat, depthFormat, sampleCount}`，提供 `getHash()`

Core 层**不**引用 `VkFormat`。Backend（`src/backend/vulkan/details/`）提供 `toVkFormat(ImageFormat) → VkFormat` 映射。

## 典型用法

```cpp
#include "core/scene/frame_graph.hpp"
#include "core/scene/scene.hpp"
#include "core/scene/pass.hpp"

using namespace LX_core;

auto scene = Scene::create(renderable);

FrameGraph frameGraph;
frameGraph.addPass(FramePass{
    Pass_Forward,
    RenderTarget{ImageFormat::BGRA8, ImageFormat::D32Float, 1},
    {}
});

// 一次调用填满所有 pass 的 queue
frameGraph.buildFromScene(*scene);

// 汇总所有去重后的 PipelineBuildInfo 交给 backend preload
auto buildInfos = frameGraph.collectAllPipelineBuildInfos();
pipelineCache.preload(buildInfos, renderPassHandle);

// 每帧绘制时直接遍历
for (auto &pass : frameGraph.getPasses()) {
    for (auto &item : pass.queue.getItems()) {
        auto &pipeline = resourceManager.getOrCreateRenderPipeline(item);
        cmd->bindPipeline(pipeline);
        cmd->bindResources(resourceManager, pipeline, item);
        cmd->drawItem(item);
    }
}
```

## 调用关系

```
VulkanRenderer::initScene(scene)
  │
  ├── m_frameGraph.addPass(FramePass{Pass_Forward, defaultForwardTarget(), {}})
  │
  ▼
FrameGraph::buildFromScene(scene)
  │ for each pass in m_passes:
  │   pass.queue.buildFromScene(scene, pass.name)
  │     │
  │     ├── clearItems()
  │     ├── sceneResources = scene.getSceneLevelResources()   // camera UBO + light UBO
  │     │
  │     ├── for each renderable in scene.getRenderables():
  │     │     if (!renderable->supportsPass(pass)) continue;
  │     │     item = makeItemFromRenderable(renderable, pass)
  │     │     item.descriptorResources += sceneResources   // 末尾追加
  │     │     m_items.push_back(item)
  │     │
  │     └── sort()   // 按 PipelineKey 稳定
  │
  ▼
FrameGraph::collectAllPipelineBuildInfos()
  │ 对每个 pass：queue.collectUniquePipelineBuildInfos() // 本 queue 内按 key 去重
  │ 全局再按 key 去重一次
  │
  ▼
PipelineCache::preload(buildInfos, renderPass)

─────── 每帧运行期 ───────

VulkanRenderer::draw()
  │
  └── for each FramePass in m_frameGraph.getPasses():
        for each RenderingItem in pass.queue.getItems():
          pipeline = resourceManager.getOrCreateRenderPipeline(item)  // 预构建命中
          cmd->bindPipeline(pipeline)
          cmd->bindResources(resourceManager, pipeline, item)
          cmd->drawItem(item)
```

## 注意事项

- **`RenderQueue::buildFromScene` 是唯一 item 构造入口**: 所有 `RenderingItem` 都从它产生，无论调用方是 `FrameGraph::buildFromScene`、backend 的 `initScene` 还是测试 helper `firstItemFromScene`。保证 scene-level 资源合并 / pass-mask 过滤 / sort 的行为一致。
- **Pass mask 过滤**: `IRenderable::supportsPass(pass)` 默认实现 `(getPassMask() & passFlagFromStringID(pass)) != 0`。不匹配的 renderable 不进入 queue，对 `collectUniquePipelineBuildInfos` 也不贡献 `PipelineBuildInfo`。
- **Scene-level UBO 在 queue 层合并**: camera UBO 和 directional light UBO 由 `Scene::getSceneLevelResources()` 集中提供，`RenderQueue::buildFromScene` 合并到每个 item 末尾。backend 的 `initScene` / `draw` **不** 做任何额外注入。
- **全局去重发生两次**: `RenderQueue::collectUniquePipelineBuildInfos()` 在 queue 内部去重，`FrameGraph::collectAllPipelineBuildInfos()` 再跨 queue 全局去重一次。相同 `PipelineKey` 只产生一个 `PipelineBuildInfo`。
- **幂等重建**: `buildFromScene` 内部调 `clearItems()`，多次调用不会累加 item。
- **`RenderTarget` 目前不参与 `PipelineKey`**: 只有一个 forward pass 时没必要；REQ-009（规划中）将让 Camera 持有 `RenderTarget`，backend `initScene` 从 device 派生 swapchain target 填入默认值。
- **排序是稳定的**: `RenderQueue::sort()` 对相同 key 的 item 保持插入顺序，避免因排序改变语义相关的 draw order。

## 测试

- `src/test/integration/test_frame_graph.cpp` — single-pass build、多 renderable 收集、跨 pass 去重、pass-mask 过滤（`testPassMaskFilterExcludesNonMatching`）、幂等重建（`testMultiPassRebuildIsIdempotent`）、`passFlagFromStringID` smoke（`testPassFlagFromStringIDSmoke`）
- `src/test/integration/test_pipeline_build_info.cpp` — `fromRenderingItem` 的 backend-agnostic 契约
- `src/test/integration/scene_test_helpers.hpp` — `firstItemFromScene(scene, pass)` 共享 helper

## 延伸阅读

- `openspec/specs/frame-graph/spec.md` — `FramePass` / `FrameGraph` / `RenderQueue::buildFromScene` / `IRenderable::supportsPass` / `Scene::getSceneLevelResources` 的所有 normative 要求
- `openspec/specs/pipeline-build-info/spec.md` — 下游的 `PipelineBuildInfo` 派生逻辑
- `openspec/specs/pipeline-cache/spec.md` — `preload` 契约
- `notes/subsystems/scene.md` — Scene 的数据容器角色
- 归档: `openspec/changes/archive/2026-04-14-frame-graph-drives-rendering/` — `FrameGraph` 真正驱动渲染路径的落地
- 归档: `openspec/changes/archive/2026-04-13-pipeline-prebuilding/` — `FrameGraph` 基础设施的首次引入
