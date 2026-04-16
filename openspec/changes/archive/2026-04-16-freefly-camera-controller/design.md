## Context

`ICameraController` 抽象和 `OrbitCameraController` 已在 REQ-015 落地。本设计新增第二个具体实现 `FreeFlyCameraController`，用于大场景内部的 FPS 风格漫游。

已有资产：
- `ICameraController::update(Camera&, const IInputState&, float dt)` — 统一签名
- `MockInputState` — 可写测试辅助
- `KeyCode` 枚举包含 W/A/S/D/Space/LShift/LCtrl
- `MouseButton::Right` 用于触发 mouse look

## Goals / Non-Goals

**Goals:**
- 实现 FPS 风格键盘移动 + 鼠标 look
- 位移严格按 dt 缩放，对角线归一化
- LCtrl 加速

**Non-Goals:**
- 不做鼠标锁定 / SDL 相对鼠标模式
- 不做手柄、action map
- 不做加速度、阻尼、惯性
- 不做碰撞检测

## Decisions

### D1: Controller 内部维护 position 和欧拉角

与 Orbit 用 (target, distance, yaw, pitch) 类似，FreeFly 用 (position, yaw, pitch)。每帧从欧拉角计算 forward/right，然后写回 Camera。

**理由**: FPS 控制器的状态就是"人站在哪、朝哪看"，这是最自然的参数化。Camera 只是输出目标。

### D2: 右键按住才启用 mouse look

不按右键时鼠标移动不影响视角。这样在没有鼠标锁定的情况下，UI 交互（将来的 ImGui）不会与 look 冲突。

### D3: 对角线移动归一化

W+D 同按时，将位移向量归一化后再乘速度，避免对角线 sqrt(2) 倍速。归一化前检查长度 > 0 避免除零。

### D4: yaw 默认 180 度

`yaw=180` 时 forward 指向 -Z，与 Camera 默认 target `(0,0,-1)` 一致。这样从 FreeFly 默认状态切换到固定相机不会突变。

### D5: forward 计算公式

```
forward.x = -cos(pitch) * sin(yaw)
forward.y = -sin(pitch)
forward.z = -cos(pitch) * cos(yaw)
right = forward.cross(worldUp).normalized()  // worldUp = (0,1,0)
```

与 Orbit 的球坐标一致但方向相反（Orbit 算 eye-to-target 偏移，FreeFly 算视线方向）。

## Risks / Trade-offs

- **[无鼠标锁定导致光标飘出窗口]** → Phase 1 可接受，REQ-017/019 再处理
- **[dt=0 时无位移]** → 鼠标 look 仍正常，符合需求约束
