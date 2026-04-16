# REQ-016: FreeFlyCameraController（FPS 风格自由飞行相机）

## 背景

REQ-015 提供的 OrbitCameraController 适合"围着一个模型转"，但调试 Sponza 这类**大型场景**时需要在内部走动观察阴影边缘 / 光照衰减 / culling 行为，这是 OrbitCamera 做不到的 —— 必须有 FPS 风格的 **自由飞行相机**：WASD 控制位移、鼠标右键按住时进入鼠标 look。

[Phase 2 REQ-208](../../notes/roadmaps/phase-2-foundation-layer.md) 规划了完整版（带 ActionMap / 手柄支持），本 REQ 是它的**最小可用前置版本**：

- 写死 WASD / Space / LShift / 鼠标右键 的硬编码 binding（不依赖 ActionMap）
- 不支持手柄
- 复用 REQ-015 引入的 `ICameraController` 接口

## 目标

1. `core/scene/freefly_camera_controller.hpp` 实现 `ICameraController`
2. WASD 控制水平位移，Space / LShift 控制垂直
3. 鼠标右键按住时锁定鼠标视角（mouse look）
4. 移动速度按 deltaTime 缩放（依赖 REQ-014 的 `Clock`）
5. LCtrl 按住时进入"加速模式"（移动速度 ×4）

## 需求

### R1: `FreeFlyCameraController` 类

新建 `src/core/scene/freefly_camera_controller.hpp` + `.cpp`：

```cpp
#pragma once
#include "core/scene/camera_controller.hpp"

namespace LX_core {

class FreeFlyCameraController : public ICameraController {
public:
  explicit FreeFlyCameraController(Vec3f startPos = {0, 0, 5},
                                   float yawDeg = 180.0f,
                                   float pitchDeg = 0.0f);

  void update(Camera &camera, const IInputState &input, float dt) override;

  // 调试 / DebugPanel 用
  Vec3f getPosition() const { return m_position; }
  float getYawDeg() const { return m_yawDeg; }
  float getPitchDeg() const { return m_pitchDeg; }
  void  setPosition(Vec3f p) { m_position = p; }
  void  setYawDeg(float y) { m_yawDeg = y; }
  void  setPitchDeg(float p) { m_pitchDeg = p; }

  // 灵敏度参数
  float moveSpeedPerSecond     = 4.0f;   // 单位/秒
  float boostMultiplier        = 4.0f;   // LCtrl 按住时
  float lookSpeedDegPerPixel   = 0.15f;  // 鼠标 look
  float minPitchDeg            = -89.0f;
  float maxPitchDeg            = 89.0f;

private:
  Vec3f m_position;
  float m_yawDeg;
  float m_pitchDeg;
};

}
```

`update` 实现要点：

1. **鼠标 look**：仅当 `input.isMouseButtonDown(Right)` 为 true 时
   - `m_yawDeg   -= input.getMouseDelta().x * lookSpeedDegPerPixel`
   - `m_pitchDeg -= input.getMouseDelta().y * lookSpeedDegPerPixel`
   - clamp pitch
2. 计算 forward / right / up 向量：
   - `yaw = radians(m_yawDeg)`, `pitch = radians(m_pitchDeg)`
   - `forward = (cos(pitch)*sin(yaw), sin(pitch), cos(pitch)*cos(yaw))`
   - `right   = normalize(cross(forward, world_up))`
   - `up_local = normalize(cross(right, forward))`（世界 up 是 (0,1,0)，做 free-fly 时本地 up 仍接近世界 up）
3. **键盘位移** —— 累计 movement 向量：
   - `W` → `+forward`
   - `S` → `-forward`
   - `D` → `+right`
   - `A` → `-right`
   - `Space` → `+world_up`
   - `LShift` → `-world_up`
4. 速度：`speed = moveSpeedPerSecond * (input.isKeyDown(LCtrl) ? boostMultiplier : 1.0f)`
5. `m_position += movement_normalized * speed * dt`
   - 注意：归一化 movement 是为了对角线移动不超速（W+D 比 W 单按快 √2）
6. 写回 camera：
   - `camera.position = m_position`
   - `camera.target   = m_position + forward`
   - `camera.up       = (0, 1, 0)`

`world_up = (0, 1, 0)` 写在文件内匿名 namespace 常量。

### R2: 鼠标右键按下时锁定鼠标？

**本 REQ 不做 SDL 鼠标锁定（`SDL_SetWindowRelativeMouseMode`）**，原因：

- 鼠标锁定会让 ImGui（REQ-017）丢失鼠标位置，需要协调
- 锁定 / 解锁的状态机要和 input state 协调，本 REQ 不动 input state
- Phase 1 调试场景里"鼠标右键按住时鼠标在屏幕里也能看到"是可接受的妥协

留 TODO 注释，REQ-017 / REQ-019 决定是否拉这个 feature。

### R3: 单元测试

新建 `src/test/integration/test_freefly_camera_controller.cpp`：

```cpp
TEST(FreeFly, w_key_moves_forward) {
  Camera cam(ResourcePassFlag::Forward);
  FreeFlyCameraController ctrl({0, 0, 0}, /*yaw*/0.0f, /*pitch*/0.0f);
  // yaw=0,pitch=0 → forward = (0,0,1)
  MockInputState input;
  input.setKeyDown(KeyCode::W, true);
  ctrl.update(cam, input, 1.0f);  // dt = 1s
  EXPECT_NEAR(cam.position.z, ctrl.moveSpeedPerSecond, 1e-3f);
}

TEST(FreeFly, mouse_look_only_with_right_button) {
  Camera cam(ResourcePassFlag::Forward);
  FreeFlyCameraController ctrl({0, 0, 0}, 0, 0);
  MockInputState input;
  input.setMouseDelta({100, 0});
  ctrl.update(cam, input, 0.016f);
  EXPECT_FLOAT_EQ(ctrl.getYawDeg(), 0.0f);  // 没按右键，yaw 不变

  input.setMouseButtonDown(MouseButton::Right, true);
  ctrl.update(cam, input, 0.016f);
  EXPECT_LT(ctrl.getYawDeg(), 0.0f);        // 鼠标向右→yaw 减
}

TEST(FreeFly, diagonal_movement_does_not_exceed_speed) {
  Camera cam(ResourcePassFlag::Forward);
  FreeFlyCameraController ctrl({0, 0, 0}, 0, 0);
  MockInputState input;
  input.setKeyDown(KeyCode::W, true);
  input.setKeyDown(KeyCode::D, true);
  ctrl.update(cam, input, 1.0f);
  float dist = std::hypot(cam.position.x, cam.position.z);
  EXPECT_NEAR(dist, ctrl.moveSpeedPerSecond, 1e-3f);
}

TEST(FreeFly, boost_multiplies_speed) {
  Camera cam(ResourcePassFlag::Forward);
  FreeFlyCameraController ctrl({0, 0, 0}, 0, 0);
  MockInputState input;
  input.setKeyDown(KeyCode::W, true);
  input.setKeyDown(KeyCode::LCtrl, true);
  ctrl.update(cam, input, 1.0f);
  EXPECT_NEAR(cam.position.z, ctrl.moveSpeedPerSecond * ctrl.boostMultiplier, 1e-3f);
}
```

## 修改范围

| 文件 | 改动 |
|---|---|
| `src/core/scene/freefly_camera_controller.hpp` | 新增 |
| `src/core/scene/freefly_camera_controller.cpp` | 新增 |
| `src/core/CMakeLists.txt` | 把新文件加进 sources |
| `src/test/integration/test_freefly_camera_controller.cpp` | 新增 |
| `src/test/integration/CMakeLists.txt` | 注册新测试 |

## 测试

见 R3。

## 边界与约束

- **不依赖** ActionMap —— Phase 2 REQ-205 之后再迁移
- **不做** 手柄 —— Phase 2 REQ-204
- **不做** 鼠标锁定 / 隐藏 —— 留 TODO，REQ-017/019 评估
- **不做** 加速度 / 阻尼模型 —— 一阶速度足够调试用
- 控制器**不**调 `camera.updateMatrices()`，由 caller 控制 update 顺序

## 依赖

- **REQ-012**（必需）：`IInputState` 接口、`KeyCode::W/A/S/D/Space/LShift/LCtrl`、`MouseButton::Right`
- **REQ-014**（必需）：deltaTime 由 `Clock` 提供
- **REQ-015**（必需）：复用 `ICameraController` 抽象基类与 `MockInputState` 测试 helper

## 下游

- **REQ-019**：demo_scene_viewer 用 F2 在 Orbit / FreeFly 之间切换
- **REQ-018**：DebugPanel 显示 / 编辑 `position` / `yawDeg` / `pitchDeg` / `moveSpeedPerSecond`
- **Phase 2 REQ-208**：扩展为带 ActionMap + 手柄的完整版

## 实施状态

2026-04-16 核查结果：未开始。
