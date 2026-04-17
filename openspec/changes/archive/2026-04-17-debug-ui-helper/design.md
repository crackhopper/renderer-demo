## Context

REQ-017 已经把 ImGui SDL3+Vulkan 接线、`VulkanRenderer::setDrawUiCallback(std::function<void()>)` 入口、SDL 事件 forward 都落地。调用方在回调里可以直接用 `ImGui::Begin` / `ImGui::SliderFloat` / `ImGui::ColorEdit3`，这是"最硬核但最啰嗦"的路径。

项目里的引擎类型并不是 POD `float[N]`：`Vec3f` / `Vec4f` 是 `VecBase<Derived, T, N>` 的子类，内部用 `union { T data[N]; struct { T x, y, z[, w]; }; }` 存储，`StringID` 则是封装了 `uint32_t id` 的结构体，可读字符串要通过 `GlobalStringTable::get().getName(id)` 反查。`DirectionalLight` 没有高层 `direction/intensity` 字段，只有 `ubo->param.dir` / `ubo->param.color` 两个 `Vec4f`，修改之后必须 `ubo->setDirty()` 才能被 `VulkanResourceManager::syncResource` 捡到。

demo（REQ-019）很快会在 UI callback 里反复写 FPS / 相机 / 方向光 面板，如果每个 demo 都自己桥 `Vec3f → float[3]`、自己拼 `std::to_string(id)` 再贴到 `ImGui::LabelText`，会快速积累样板代码并让风格发散。本 REQ 的目标仅限于：抽一组无状态、可组合、可与原生 ImGui 混用的 helper，放在 `infra/gui/`。

## Goals / Non-Goals

**Goals:**
- 在 `src/infra/gui/debug_ui.{hpp,cpp}` 暴露一组命名空间 `LX_infra::debug_ui` 的薄 helper
- 基础桥接：`Vec3f`/`Vec4f` 拖动、滑条、颜色编辑、label 展示、`StringID` 解析
- 统一 panel / section 容器默认风格
- 提供 `renderStatsPanel(Clock&)` / `cameraPanel(title, Camera&)` / `directionalLightPanel(title, DirectionalLight&)` 三个开箱即用面板
- helper 可任意嵌入原生 ImGui 代码中；状态由调用方持有（开关、持久显示偏好等）

**Non-Goals:**
- 不实现 `materialPanel()` 或任意自动反射编辑器
- 不做 scene graph inspector
- 不引入新的 GUI framework / DSL / 响应式 panel 描述器
- 不做 docking / multi-viewport / 表格 DSL / 自定义字体
- 不把 ImGui 依赖泄漏到 `core`（helper 只在 `LX_infra` 生存）
- 不改动 `gui.hpp` / `imgui_gui.cpp` 既有公开接口
- 不做像素级截图 / headless CI ImGui 渲染验证

## Decisions

### D1: 命名空间 `LX_infra::debug_ui`，文件放在 `src/infra/gui/`

**选择**：所有 helper 放在 `namespace LX_infra::debug_ui`，对应源码在 `src/infra/gui/debug_ui.{hpp,cpp}`。

**替代方案**：
- 放到 `namespace infra`（与现有 `infra::Gui` 一致）→ 与本项目主流命名空间 `LX_infra` 不一致；`Gui` 命名空间是遗留，不应继续扩散
- 单独放到 `src/infra/debug_ui/` → helper 的"薄"属性不够支撑一个子目录；与 ImGui 一起归在 `gui/` 更自然

**理由**：符合 `LX_core` / `LX_infra` 约定；helper 与 `infra/gui/` 已有的 ImGui 封装同层；不拉高模块边界。

### D2: `Vec3f` / `Vec4f` 桥接复用 `value.data[0]` 指针，加静态断言保护

**选择**：

```cpp
static_assert(sizeof(Vec3f) == 3 * sizeof(float), "Vec3f must be tightly packed");
static_assert(offsetof(Vec3f, data) == 0, "Vec3f data must be at offset 0");
bool dragVec3(const char* label, Vec3f& v, float speed, float min, float max) {
  return ImGui::DragFloat3(label, v.data, speed, min, max);
}
```

**替代方案**：
- 每次拷贝到临时 `float[3]`，编辑后写回 → 多一次内存拷贝，且两个方向都要写
- 要求 `VecBase` 暴露 `raw()` 指针 → 要改 `core/math/vec.hpp`，扩大了改动面

**理由**：`Vec3f` / `Vec4f` 用 union 让 `data` 字段与结构体起始地址对齐；当前布局下 `sizeof(VecN) == N * sizeof(float)` 且 `data` 在 offset 0。静态断言能在未来 math 类型变更时立刻打断编译，而不是生成静默的内存错乱。helper 内部直接复用指针，零拷贝。

### D3: `labelStringId` 通过 `GlobalStringTable::get().getName(id.id)` 查名

**选择**：`labelStringId(label, StringID)` 内部调 `GlobalStringTable::get().getName(value.id)` 拿到 `const std::string&`，然后走 `ImGui::LabelText("%s", "%s", label, name.c_str())`。

**替代方案**：要求调用方先自行解析成字符串再传入 → 把"字符串化"这层样板推回给 demo，与本 REQ 目标相反。

**理由**：单 `StringID` 的读侧是 O(1) unordered_map 查找；`debug_ui` 面板一帧触发次数很小，不需要缓存。

### D4: `cameraPanel` 不隐式 `updateMatrices()`

**选择**：helper 只编辑 `camera.position/target/up/fovY/aspect/nearPlane/farPlane` 字段；**不**在内部调 `camera.updateMatrices()`。

**替代方案**：helper 自己 update → 把隐藏副作用塞进 UI 代码，且一帧内多个 panel 编辑同一个相机时会重复调用。

**理由**：REQ 原文明确要求"是否以及何时调用 updateMatrices 由调用方负责"。调用方（demo / engine loop）更清楚一帧内什么时机适合更新矩阵（通常是所有 UI 结束后统一刷一次）。helper 的 return value 仅是 "是否有字段被改动"（`bool` 或聚合 `return changed`），调用方可据此决定是否 update。

### D5: `directionalLightPanel` 直接写 `ubo->param.{dir,color}`，修改后 `setDirty()`

**选择**：

```cpp
void directionalLightPanel(const char* title, DirectionalLight& light) {
  bool changed = false;
  // dir 是 Vec4f，但语义上前 3 分量才是方向；用 dragVec4 允许同时编辑 w
  // （或用 dragVec3 + xyz；design 上用 dragVec4 给出更直白的原始视图）
  changed |= dragVec4("dir", light.ubo->param.dir, 0.01f);
  changed |= colorEdit4("color", light.ubo->param.color);
  if (changed) {
    light.ubo->setDirty();
  }
}
```

**替代方案**：封装一个"高层光模型"（direction: Vec3f, intensity: float, color: Vec3f）→ 超出"薄 helper"范围；`DirectionalLight` 当前没有这套结构，本 REQ 明确禁止自说自话。

**理由**：使用真实数据布局；`setDirty()` 保持单次写入——即便一帧内改多个字段也只 dirty 一次，对 `VulkanResourceManager` 的影响最小。

### D6: Panel 容器默认策略

**选择**：
- `beginPanel(title)`：`ImGui::SetNextWindowPos({8, 8}, ImGuiCond_FirstUseEver)` + `ImGui::SetNextWindowSize({320, 400}, ImGuiCond_FirstUseEver)` + `ImGui::Begin(title)`，返回 `bool` 表示 window 是否可见
- `endPanel()`：无条件 `ImGui::End()`（ImGui 规则：Begin 返回 false 仍需调 End）
- `beginSection(title)`：`ImGui::CollapsingHeader(title)` 返回 bool
- `endSection()`：no-op（CollapsingHeader 不需要匹配调用；保留该 API 是为了对称与未来改成 TreeNode 时不破坏调用点）
- `separatorText(label)`：透传 `ImGui::SeparatorText`

**替代方案**：
- 用 `ImGui::TreeNode` 做 section → 需要对称 `TreePop`；beginSection/endSection 对称更合意，但每次都必须进入 TreeNode 分支会显得啰嗦
- Begin 返回 true 才必须配 End → 与 ImGui 实际约定不符（Begin 无论返回值都必须 End）

**理由**：默认 `FirstUseEver` 让位置/大小第一次生效，之后由用户拖动；对新 demo 而言"立即可见且不重叠主视口左上"是最有用的默认。`endSection` 哑函数保留对称性，成本为零。

### D7: `beginPanel` 返回 bool，`endPanel` 无条件调用

**选择**：调用方写：
```cpp
if (beginPanel("Stats")) {
  renderStatsPanel(clock);
}
endPanel();
```

**替代方案**：把 `endPanel` 做成 RAII `PanelScope` → helper 就有了状态；且调用方需要引入 block scope，打断了 "想到哪写到哪" 的 ImGui 编程风格。

**理由**：维持与 ImGui 一贯的"Begin/End 总是成对，Begin 返回 false 表示内容被折叠/剪裁，仍要 End"约定；调用模式可直接类比 `ImGui::Begin/End`。

### D8: 测试只做链接级 + 可选的 ImGui context smoke

**选择**：新建 `test_debug_ui_smoke.cpp`：
1. 取每个 helper 的函数地址存到 `void*` 数组，断言非零——这相当于链接级存在性检查
2. 若 `ImGui::CreateContext()` 可用（不需要 Vulkan / 不需要 window），执行一次 `ImGui::NewFrame` → 调 helper → `ImGui::EndFrame`，断言不崩溃
3. 若 CI 环境不允许创建 context（极少见，ImGui 纯 CPU 创建 context 不需要图形上下文），则只做链接级

**替代方案**：
- GoogleTest + 截图验证 → 超出 REQ 范围
- 完全跳过测试 → 违反 REQ-018 R6

**理由**：REQ 明确"测试目标以编译/链接与极薄行为验证为主"，且"允许只做链接级验证"。核心价值是捕获"有人删了 helper 但没更新 demo / 后续 REQ 依赖缺失"这类回归。

## Risks / Trade-offs

- **[Vec3f / Vec4f 内存布局变更]** → helper 会直接退化（浮点指针错位，debug 面板里看到乱码/崩溃）；缓解：D2 的 `static_assert` 会让编译期立刻打断，迫使维护者一起修 helper
- **[`beginSection/endSection` 若日后换成 TreeNode 就必须配对 pop]** → 现在 `endSection` 是 no-op，切换实现时所有调用点已经有 `endSection()`，改动只在 `.cpp` 内部；短期不是问题
- **[`labelStringId` 查不到 name（id 未注册）]** → `GlobalStringTable::getName` 若返回空字符串，helper 显示 "(empty)" 兜底，避免 UI 看上去空白而误以为 bug
- **[CI 无 ImGui context 时 smoke 跳过]** → 保留 "链接级" 兜底可接受；若未来 CI 支持 context，可追加真实 smoke
- **[一帧内多处 panel 编辑同一 `DirectionalLight`]** → 多次 `setDirty()` 无害（只是把同一个 flag 置位）；不会引发重复上传
- **[`cameraPanel` 改了字段但调用方忘了 `updateMatrices()`]** → 视图矩阵下一帧才更新；这是接受的显式代价（REQ 明确要求）。可以考虑在 helper 返回值里带 `bool changed`，demo 据此决定是否 update

## Migration Plan

1. 新增 `src/infra/gui/debug_ui.{hpp,cpp}`，包含 D2 的 static_assert、基础桥接、panel 容器
2. 在同一文件里实现 `renderStatsPanel` / `cameraPanel` / `directionalLightPanel`
3. 把 `gui/debug_ui.cpp` 追加到 `src/infra/CMakeLists.txt` 的 `INFRA_SOURCES`
4. 新增 `test_debug_ui_smoke.cpp`，注册到 `src/test/CMakeLists.txt`
5. `cmake --build build` + 跑 smoke 测试验证

每一步都可独立 build；不需要 flag 门控；既有 REQ-017 的 `Gui` / `VulkanRenderer` 接线零变更。

## Open Questions

- `cameraPanel` / `directionalLightPanel` 的返回值是 `void` 还是 `bool changed`？当前 REQ 文本是 `void`；为了让调用方正确决定是否 `updateMatrices()`，实现阶段可能会选 `bool`——此改动不破坏 spec，只是更贴合使用场景。→ 任务阶段选定，spec 允许 `void`。
