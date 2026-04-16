# REQ-015: OrbitCameraController（轨道相机控制器）

## 背景

`src/core/scene/camera.hpp:46-114` 的 `Camera` 类是纯数据 —— position / target / up / fov 等字段，每帧调 `updateMatrices()`，**没有任何输入交互逻辑**。

调试 PBR / IBL / shadow 的最直观方式是"用鼠标围着模型转 + 滚轮缩放"，这是 modeling viewer 风格的 **轨道相机（orbit camera）** —— 也是 Blender / Maya / glTF Viewer / Sketchfab 的默认交互。本 REQ 把这个控制器抽出来作为独立类，与 `Camera` 解耦。

为什么不直接把 controller 塞进 `Camera`：

- `Camera` 是一个数据结构，序列化语义清晰；controller 含 input 依赖会污染
- REQ-016 还有 FreeFly 控制器，运行时可能切换；同一个 `Camera` 被两种 controller 操作
- 对应 [Phase 2 REQ-208](../../notes/roadmaps/phase-2-foundation-layer.md) 把 controller 与 input system 解耦的方向

本需求**只**实现 OrbitCameraController，不动 `Camera` 类签名。

## 目标

1. `core/scene/camera_controller.hpp` 提供一个抽象基类 `ICameraController`
2. `OrbitCameraController` 是其第一个具体实现
3. 鼠标左键拖拽 → 围绕 target 旋转
4. 鼠标滚轮 → 沿视线方向 zoom in/out
5. 鼠标右键拖拽 → 平移 target（pan）
6. 不依赖 `Camera` 类的内部修改

## 需求

### R1: `ICameraController` 抽象基类

新建 `src/core/scene/camera_controller.hpp`：

```cpp
#pragma once
#include "core/input/input_state.hpp"
#include "core/scene/camera.hpp"

namespace LX_core {

class ICameraController {
public:
  virtual ~ICameraController() = default;

  /// 每帧调用一次。controller 根据 input state 修改 camera 的 position / target / up，
  /// 但不调 camera->updateMatrices() —— 那是 caller 的职责（统一更新点）。
  virtual void update(Camera &camera, const IInputState &input, float dt) = 0;
};

using CameraControllerPtr = std::shared_ptr<ICameraController>;

}
```

设计取舍：

- `update` 接 `Camera &` 而不是 `CameraPtr` —— controller 不持有 camera 生命周期
- `update` 接 `const IInputState &` —— controller 不持有 input state，由 caller 注入
- `update` 接 `float dt` —— Orbit 不强需要 dt（鼠标 delta 已经是 per-event 量），但 FreeFly 需要；统一签名

### R2: `OrbitCameraController` 实现

新建 `src/core/scene/orbit_camera_controller.hpp` + `.cpp`：

```cpp
namespace LX_core {

class OrbitCameraController : public ICameraController {
public:
  /// 围绕给定 target 的初始 distance / yaw / pitch。
  /// 调用 update() 时若 camera->target 与构造参数不一致，以构造参数为准
  /// （让 controller 完全 own 视锥状态，避免外部直接改 target 又改 controller 造成冲突）。
  OrbitCameraController(Vec3f target = {0, 0, 0},
                        float distance = 5.0f,
                        float yawDeg = 0.0f,
                        float pitchDeg = 20.0f);

  void update(Camera &camera, const IInputState &input, float dt) override;

  // 调试 / DebugPanel 用的 getter/setter
  Vec3f getTarget() const { return m_target; }
  void  setTarget(Vec3f t) { m_target = t; }
  float getDistance() const { return m_distance; }
  void  setDistance(float d);
  float getYawDeg() const { return m_yawDeg; }
  float getPitchDeg() const { return m_pitchDeg; }

  // 灵敏度参数
  float rotateSpeedDegPerPixel = 0.4f;  // 鼠标左键拖拽
  float panSpeedPerPixel       = 0.005f; // 鼠标右键拖拽（按 distance 等比）
  float zoomSpeedPerWheel      = 0.15f;  // 滚轮一格的缩放比例
  float minDistance            = 0.5f;
  float maxDistance            = 200.0f;
  float minPitchDeg            = -89.0f;
  float maxPitchDeg            = 89.0f;

private:
  Vec3f m_target;
  float m_distance;
  float m_yawDeg;
  float m_pitchDeg;
};

}
```

`update` 实现要点：

1. 读 `input.getMouseDelta()` —— 像素 delta，本帧累计
2. 读 `input.getMouseWheelDelta()`
3. 若 `input.isMouseButtonDown(Left)`：
   - `m_yawDeg   -= delta.x * rotateSpeedDegPerPixel`
   - `m_pitchDeg -= delta.y * rotateSpeedDegPerPixel`
   - clamp pitch 到 `[minPitchDeg, maxPitchDeg]`
4. 若 `input.isMouseButtonDown(Right)`：
   - 计算 camera right / up 向量（基于当前 yaw/pitch）
   - `m_target += -right * delta.x * panSpeedPerPixel * m_distance`
   - `m_target += +up    * delta.y * panSpeedPerPixel * m_distance`
5. 滚轮：`m_distance *= (1.0f - wheel * zoomSpeedPerWheel)`，clamp 到 `[minDistance, maxDistance]`
6. 由 `(m_target, m_distance, m_yawDeg, m_pitchDeg)` 计算 camera position：
   - `yaw = radians(m_yawDeg)`, `pitch = radians(m_pitchDeg)`
   - `dir = (cos(pitch)*sin(yaw), sin(pitch), cos(pitch)*cos(yaw))`
   - `camera.position = m_target + dir * m_distance`
   - `camera.target   = m_target`
   - `camera.up       = (0, 1, 0)`

注：当 pitch 接近 ±90° 时 up 与 dir 几乎共线，clamp 已经避开。

### R3: 单元测试

新建 `src/test/integration/test_orbit_camera_controller.cpp`：

构造一个 `Camera` + `OrbitCameraController` + `DummyInputState`，喂手工事件，断言 camera 状态：

```cpp
TEST(OrbitController, default_position_is_in_front_of_target) {
  Camera cam(ResourcePassFlag::Forward);
  OrbitCameraController ctrl({0, 0, 0}, 5.0f, /*yaw*/0, /*pitch*/0);
  DummyInputState input;
  ctrl.update(cam, input, 0.016f);
  EXPECT_NEAR(cam.position.z, 5.0f, 1e-4f);
  EXPECT_NEAR(cam.position.x, 0.0f, 1e-4f);
  EXPECT_NEAR(cam.position.y, 0.0f, 1e-4f);
}

// 类似的：
//   - rotate_with_left_drag (用一个能被 set 的 mock input state)
//   - clamp_pitch_at_limits
//   - zoom_clamps_distance
```

由于 `DummyInputState` 是 const-only stub，本 REQ 还要新增一个 `MockInputState`（同样在 `core/input/`，仅 testing 用）允许 setter 写入字段。**这个 MockInputState 是必要的副产物**，不另开 REQ。

### R4: `MockInputState` testing helper

新建 `src/core/input/mock_input_state.hpp`：

```cpp
namespace LX_core {

class MockInputState : public IInputState {
public:
  void setKeyDown(KeyCode k, bool down) { m_keys[(size_t)k] = down; }
  void setMouseButtonDown(MouseButton b, bool down) { m_buttons[(size_t)b] = down; }
  void setMousePosition(Vec2f p) { m_pos = p; }
  void setMouseDelta(Vec2f d) { m_delta = d; }
  void setMouseWheelDelta(float w) { m_wheel = w; }

  bool   isKeyDown(KeyCode k) const override { return m_keys[(size_t)k]; }
  bool   isMouseButtonDown(MouseButton b) const override { return m_buttons[(size_t)b]; }
  Vec2f  getMousePosition() const override { return m_pos; }
  Vec2f  getMouseDelta() const override { return m_delta; }
  float  getMouseWheelDelta() const override { return m_wheel; }
  void   nextFrame() override { m_delta = {0, 0}; m_wheel = 0; }

private:
  std::array<bool, (size_t)KeyCode::Count>     m_keys{};
  std::array<bool, (size_t)MouseButton::Count> m_buttons{};
  Vec2f m_pos{0, 0};
  Vec2f m_delta{0, 0};
  float m_wheel = 0;
};

}
```

注意：这是测试 fixture，header-only，不进 production build path。但放在 `core/input/` 而不是 `test/`，方便 REQ-016 / REQ-018 也复用。

## 修改范围

| 文件 | 改动 |
|---|---|
| `src/core/scene/camera_controller.hpp` | 新增 |
| `src/core/scene/orbit_camera_controller.hpp` | 新增 |
| `src/core/scene/orbit_camera_controller.cpp` | 新增 |
| `src/core/input/mock_input_state.hpp` | 新增（R4） |
| `src/core/CMakeLists.txt` | 把新文件加进 sources |
| `src/test/integration/test_orbit_camera_controller.cpp` | 新增 |
| `src/test/integration/CMakeLists.txt` | 注册新测试 |

## 测试

见 R3。

## 边界与约束

- **不修改** `Camera` 类签名 —— 保持 REQ-009 之后的状态
- **不做** 帧间插值 / smoothing —— 鼠标 delta 已经按帧累积，再 smooth 会增加延迟感
- **不做** Trackball / ArcBall —— Phase 1 以后可选
- **不做** 触屏 / 手势 —— 后续考虑
- 控制器**不**调 `camera.updateMatrices()` —— caller 在 update 后自行调，让 caller 控制 update 顺序

## 依赖

- **REQ-012**（必需）：`IInputState` 接口
- **REQ-013**（推荐，不强依赖）：真实的 SDL3 输入实现 —— 测试可以走 `MockInputState` 不依赖 SDL

## 下游

- **REQ-019**：demo_scene_viewer 默认相机控制器
- **REQ-018**：DebugPanel 显示 / 编辑 `target` / `distance` / `yawDeg` / `pitchDeg`
- **Phase 2 REQ-208**：FreeFly 完整版 + ActionMap 时回头复用同一个 `ICameraController` 接口

## 实施状态

2026-04-16 核查结果：未开始。
