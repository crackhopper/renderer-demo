# Scene

> 场景层是一个**纯数据容器**：持有可渲染对象集合（mesh + material + 可选 skeleton）、camera、directional light。`RenderingItem` 的构造不在 Scene，而在 `RenderQueue::buildFromScene(scene, pass)`——Scene 只提供 getter。
>
> 相关 spec: `openspec/specs/pipeline-key/spec.md`（`RenderingItem` 字段）、`openspec/specs/render-signature/spec.md`（`getRenderSignature(pass)` 贯通）、`openspec/specs/frame-graph/spec.md`（`RenderQueue::buildFromScene` + `IRenderable::supportsPass` + `Scene::getSceneLevelResources()`）

## 核心抽象

### `RenderingItem` (`src/core/scene/scene.hpp:14`)

一次 draw call 的完整上下文值对象：

```cpp
struct RenderingItem {
    IShaderPtr                       shaderInfo;
    ObjectPCPtr                      objectInfo;           // push constant 源
    IRenderResourcePtr               vertexBuffer;
    IRenderResourcePtr               indexBuffer;
    std::vector<IRenderResourcePtr>  descriptorResources;  // material + skeleton + camera + light
    ResourcePassFlag                 passMask;
    StringID                         pass;                 // 当前 pass 的 StringID
    PipelineKey                      pipelineKey;          // 身份
    MaterialPtr                      material;             // 用于 PipelineBuildInfo 派生
};
```

### `Scene` (`src/core/scene/scene.hpp:30`)

```cpp
class Scene {
public:
    CameraPtr                    camera;
    DirectionalLightPtr          directionalLight;

    void addRenderable(IRenderablePtr r);
    const std::vector<IRenderablePtr> &getRenderables() const;

    /// 场景级 descriptor 资源（camera UBO + light UBO）。
    /// RenderQueue::buildFromScene 在构造每个 RenderingItem 时合并到末尾，
    /// 取代任何"backend 手工注入 camera/light UBO"的老路径。
    std::vector<IRenderResourcePtr> getSceneLevelResources() const;

private:
    std::vector<IRenderablePtr>  m_renderables;
};
```

`Scene` 不再暴露任何 item factory 方法。`RenderingItem` 构造职责完整迁到 `RenderQueue`。

### `IRenderable` (`src/core/scene/object.hpp:50`)

```cpp
class IRenderable {
public:
    virtual IRenderResourcePtr getVertexBuffer() const = 0;
    virtual IRenderResourcePtr getIndexBuffer() const = 0;
    virtual std::vector<IRenderResourcePtr> getDescriptorResources() const = 0;
    virtual ResourcePassFlag getPassMask() const = 0;
    virtual IShaderPtr getShaderInfo() const = 0;
    virtual ObjectPCPtr getObjectInfo() const { return nullptr; }
    virtual StringID getRenderSignature(StringID pass) const = 0;

    /// 该 renderable 是否参与指定 pass。默认实现：
    ///   (getPassMask() & passFlagFromStringID(pass)) != 0
    /// RenderQueue::buildFromScene 用它过滤 per-pass 的 item 集合。
    virtual bool supportsPass(StringID pass) const;
};
```

### `RenderableSubMesh` (`src/core/scene/object.hpp:69`)

```cpp
struct RenderableSubMesh : public IRenderable {
    MeshPtr                       mesh;
    MaterialPtr                   material;        // MaterialInstance
    std::optional<SkeletonPtr>    skeleton;
    ObjectPCPtr                   objectPC;

    // IRenderable 接口全部 override
    // getDescriptorResources 聚合 material 资源 + skeleton UBO（若有）
    // getRenderSignature 做 compose(ObjectRender, {meshSig, skelSig})
    // supportsPass 使用默认实现（基于 material 的 passFlag）
};
```

### 场景级 UBO

- **`CameraUBO`** (`src/core/scene/camera.hpp:11`) — view / proj / eye，继承 `IRenderResource`
- **`DirectionalLightUBO`** (`src/core/scene/light.hpp:8`) — dir / color
- **`ObjectPC`** (`src/core/scene/object.hpp:15`) — push constant，128 字节缓冲，继承 `IRenderResource`

这些 UBO 由 `Scene` / `Camera` / `DirectionalLight` 持有。`Scene::getSceneLevelResources()` 返回 `{camera UBO, directional light UBO}`（顺序固定，非空 optional），`RenderQueue::buildFromScene` 自动合并到每个 `RenderingItem::descriptorResources` 末尾，backend 无需额外注入。

### `RenderQueue::buildFromScene` (`src/core/scene/render_queue.hpp:34`)

```cpp
class RenderQueue {
public:
    /// 按 pass 从 scene 构建 queue 里所有 RenderingItem：
    ///   1. clearItems()
    ///   2. 拉取 scene.getSceneLevelResources() 一次
    ///   3. 遍历 scene.getRenderables(), 跳过 !supportsPass(pass) 的
    ///   4. 为每个匹配的 renderable 构造 RenderingItem (文件内 makeItemFromRenderable helper)
    ///   5. 把 scene 级资源追加到 item.descriptorResources 末尾
    ///   6. sort() 按 PipelineKey 稳定排序
    void buildFromScene(const Scene &scene, StringID pass);
};
```

## 典型用法

```cpp
#include "core/scene/scene.hpp"
#include "core/scene/pass.hpp"
#include "core/scene/render_queue.hpp"
#include "core/scene/frame_graph.hpp"
#include "core/scene/object.hpp"

using namespace LX_core;

// 1. 构建 renderable
auto mesh     = Mesh::create(vertexBuffer, indexBuffer);
auto material = LX_infra::loadBlinnPhongMaterial();
auto skeleton = Skeleton::create({});   // 空骨骼也要构造
auto renderable = std::make_shared<RenderableSubMesh>(mesh, material, skeleton);

// 2. 场景
auto scene = Scene::create(renderable);

// 3. 配置 camera / light (Scene 构造时已创建)
scene->camera->position = {0, 0, 3};
scene->camera->target   = {0, 0, 0};
scene->camera->up       = {0, 1, 0};
scene->camera->updateMatrices();

// 4. 从 Scene 构建单 pass 的 RenderQueue
RenderQueue queue;
queue.buildFromScene(*scene, Pass_Forward);
// queue.getItems() 已填满；每个 item.pipelineKey / item.pass 已填充，
// descriptorResources 尾部已合并 camera UBO + light UBO

// 5. 多 pass 走 FrameGraph（同样的入口，FrameGraph 内部调 queue.buildFromScene）
FrameGraph frameGraph;
frameGraph.addPass({Pass_Forward, {}, {}});
frameGraph.buildFromScene(*scene);
// 每个 FramePass 的 queue 已填满
```

## 调用关系

```
VulkanRenderer::initScene(scene)
  │
  ├── m_frameGraph.addPass(FramePass{Pass_Forward, defaultForwardTarget(), {}})
  │
  ├── m_frameGraph.buildFromScene(*scene)
  │     │
  │     └── for each FramePass:
  │           pass.queue.buildFromScene(scene, pass.name)
  │             │
  │             ├── clearItems()
  │             ├── sceneResources = scene.getSceneLevelResources()   // camera + light UBO
  │             │
  │             ├── for each renderable in scene.getRenderables():
  │             │     if (!renderable->supportsPass(pass)) continue;
  │             │     item = makeItemFromRenderable(renderable, pass):
  │             │         vertexBuffer / indexBuffer / objectInfo
  │             │         descriptorResources = renderable->getDescriptorResources()
  │             │         shaderInfo / passMask / pass = ...
  │             │         if RenderableSubMesh + mesh + material:
  │             │             item.material    = sub->material
  │             │             item.pipelineKey = PipelineKey::build(
  │             │                 sub->getRenderSignature(pass),
  │             │                 sub->material->getRenderSignature(pass))
  │             │     item.descriptorResources.insert(end, sceneResources)
  │             │     m_items.push_back(item)
  │             │
  │             └── sort()   // 按 PipelineKey 稳定排序
  │
  └── preloadPipelines(m_frameGraph.collectAllPipelineBuildInfos())

VulkanRenderer::draw()
  │
  └── for each FramePass in m_frameGraph.getPasses():
        for each item in pass.queue.getItems():
          pipeline = resourceManager->getOrCreateRenderPipeline(item)
          cmd->bindPipeline(pipeline)
          cmd->bindResources(*resourceManager, pipeline, item)   // 按 binding.name 匹配
          cmd->drawItem(item)
```

## 注意事项

- **Scene 是纯数据容器**: 除了 setter/getter 和 `getSceneLevelResources()`，Scene 不暴露任何业务逻辑。item factory / pass 过滤 / 资源合并全部在 `RenderQueue::buildFromScene` 内完成。
- **`RenderingItem::material` 字段用于 PipelineBuildInfo**: `PipelineBuildInfo::fromRenderingItem(item)` 需要从 material 派生 `renderState`，所以 item 持 material 的 shared_ptr。
- **Scene UBO 不在 `IRenderable::getDescriptorResources()` 里**: `RenderableSubMesh::getDescriptorResources()` 只返回 material 资源和可选的 skeleton UBO。camera / light UBO 由 `Scene::getSceneLevelResources()` 单独暴露，`RenderQueue::buildFromScene` 合并到每个 item 末尾。这种分离让 material 的资源列表保持纯粹。
- **Push constant 通过 `ObjectPC`**: `ObjectPC : IRenderResource` 存 128 字节的 model 矩阵 + flags。`RenderableSubMesh` 构造时创建一个，由 material 的 passFlag 决定它属于哪一 pass。
- **Pass 参数贯穿**: 从 `RenderQueue::buildFromScene(scene, pass)` 开始，pass 穿过 `IRenderable::getRenderSignature(pass)` → `Mesh::getRenderSignature(pass)` → `IMaterial::getRenderSignature(pass)` → `MaterialTemplate::getRenderPassSignature(pass)`。任何一环弄丢 pass 参数都会导致 pipeline key 混乱。
- **`supportsPass` 默认实现**: `(getPassMask() & passFlagFromStringID(pass)) != 0`。`passFlagFromStringID` 在 `src/core/scene/pass.cpp` 里把 `Pass_Forward` / `Pass_Deferred` / `Pass_Shadow` 翻译成对应的 `ResourcePassFlag` 位；未知 pass 返回 0 位，被过滤掉。

## 测试

- `src/test/integration/test_vulkan_command_buffer.cpp` — 端到端路径：`firstItemFromScene` helper → bindResources → draw
- `src/test/integration/test_pipeline_identity.cpp` — `RenderQueue::buildFromScene` 产出的 item 的 pipelineKey 填充
- `src/test/integration/test_frame_graph.cpp` — 多 renderable × 多 pass 扫描、pass-mask 过滤、幂等重建
- `src/test/integration/scene_test_helpers.hpp` — `firstItemFromScene(scene, pass)` 共享 helper，封装 `RenderQueue::buildFromScene` + `getItems().front()` + 非空断言

## 延伸阅读

- `openspec/specs/frame-graph/spec.md` — `RenderQueue::buildFromScene` / `IRenderable::supportsPass` / `Scene::getSceneLevelResources` 契约
- `openspec/specs/pipeline-key/spec.md` — `RenderingItem::pipelineKey + pass` 字段契约
- `openspec/specs/render-signature/spec.md` — `IRenderable::getRenderSignature(pass)` 要求
- `notes/subsystems/frame-graph.md` — FrameGraph / FramePass / RenderQueue 组合
- `notes/subsystems/pipeline-identity.md` — `PipelineKey` 的构造
- `notes/subsystems/material-system.md` — material 如何贡献 `materialSig`
