## Why

`Clock` 已有 `tick()`/`deltaTime()`/`totalTime()`/`frameCount()`，但还缺 `smoothedDeltaTime()` 来给调试统计面板（REQ-018）提供稳定的 FPS 读数。当前 `Clock` 也没有独立测试保护既有行为。本变更只做收尾：补平滑 deltaTime 和测试，不扩展到 Phase 2 的 fixed step / timeScale / pause。

## What Changes

- `Clock` 新增 `smoothedDeltaTime()` 方法，基于 60 帧滑动平均的环形缓冲
- `tick()` 在保持现有行为的前提下，额外将后续帧的 deltaTime 写入平滑窗口
- 新增 `test_clock` 集成测试，覆盖首次 tick 零值、非零 delta、totalTime 单调递增、平滑 delta 回退和平均语义

## Capabilities

### New Capabilities
- `clock-time-system`: Clock 类的完整契约，包括既有接口和新增 `smoothedDeltaTime()`

### Modified Capabilities

## Impact

- **代码**：`src/core/time/clock.hpp`（新增声明和内部状态）、`src/core/time/clock.cpp`（平滑窗口写入和实现）
- **测试**：新增 `src/test/integration/test_clock.cpp`，注册到 `src/test/CMakeLists.txt`
- **依赖**：无外部新增依赖
