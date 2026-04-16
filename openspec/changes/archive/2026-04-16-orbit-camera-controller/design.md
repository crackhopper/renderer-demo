## Context

Camera（`src/core/scene/camera.hpp`）是纯数据 + 矩阵更新对象，不含交互逻辑。输入抽象（`IInputState`）已就绪，但只有全零的 `DummyInputState`，无法驱动交互测试。本设计引入 controller 抽象层，让输入到相机姿态的映射独立于 Camera 本体。

## Goals / Non-Goals

**Goals:**
- 提供 `ICameraController` 纯虚基类，统一 controller 签名
- 实现 `OrbitCameraController`，覆盖旋转/平移/缩放三种交互
- 提供 `MockInputState`，让控制器测试完全脱离 SDL
- 集成测试验证核心行为

**Non-Goals:**
- 不修改 Camera 类签名
- 不做 Trackball / ArcBall
- 不做触屏/手势
- 不做平滑插值、惯性或阻尼
- 不做 UI 抢占输入协调（REQ-017/019 范围）

## Decisions

### D1: Controller 不持有 Camera 也不持有 InputState

`update(Camera&, const IInputState&, float dt)` 每帧传入引用，controller 不管生命周期。

**理由**: Camera 归 Scene 管理，InputState 归 Window/Platform 管理。controller 只做映射，不拥有资源。FreeFly 等后续 controller 可复用同一签名。

**备选**: controller 持有 Camera 弱引用——增加耦合，且 Camera 切换时需重新绑定，不如直接传参简洁。

### D2: Orbit 状态用 (target, distance, yaw, pitch) 四元组

每帧从四元组反算 `camera.position`、`camera.target`、`camera.up`。

**理由**: 这是 orbit camera 的经典参数化方式，直觉可控、容易 clamp、方便调试面板暴露。

**备选**: 四元数表示——orbit 不需要万向锁保护（pitch 已 clamp），四元数增加理解成本。

### D3: Controller 不调用 `camera.updateMatrices()`

controller 只写 Camera 的空间参数（position/target/up），矩阵更新由调用方统一调度。

**理由**: 渲染循环通常在所有 controller 更新后、draw 前统一 updateMatrices。在 controller 内部调用会导致多余的矩阵计算。

### D4: MockInputState 放在 `src/core/input/`

与 DummyInputState 同级，header-only。

**理由**: MockInputState 是 IInputState 的一种实现，不应放到 test 目录下——后续 demo 程序也可能用它做脚本化输入。且 core 库不依赖 SDL，放在 core/input 不引入额外依赖。

### D5: Orbit 数学用标准球坐标

```
eye.x = target.x + distance * cos(pitch) * sin(yaw)
eye.y = target.y + distance * sin(pitch)
eye.z = target.z + distance * cos(pitch) * cos(yaw)
```

yaw=0, pitch=0 时相机在 target 正前方 (+Z 方向)。up 固定 (0,1,0)。

## Risks / Trade-offs

- **[Gimbal lock at pitch +-90]** → pitch clamp 到 [-89, 89] 度，orbit 场景下足够
- **[up 固定 (0,1,0) 限制倾斜]** → orbit viewer 不需要 roll，可接受
- **[panSpeed 依赖 distance]** → 当前用固定 panSpeedPerPixel，远距离平移会感觉慢——后续可改为 `panSpeed * distance`，但 REQ-015 明确不做 smoothing，先保持简单
