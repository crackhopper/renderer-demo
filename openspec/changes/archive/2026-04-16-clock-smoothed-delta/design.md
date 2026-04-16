## Context

`Clock`（`src/core/time/clock.hpp/.cpp`）已在 `EngineLoop::tickFrame()` 中被推进。现有接口：`tick()`、`deltaTime()`、`totalTime()`、`frameCount()`。内部使用 `std::chrono::steady_clock`，首次 tick 初始化时间点并返回零值，后续 tick 计算真实时间差。

## Goals / Non-Goals

**Goals:**
- 新增 `smoothedDeltaTime()`，60 帧滑动平均
- 补集成测试固定现有 + 新增行为

**Non-Goals:**
- 不做 fixed step accumulator、timeScale、pause
- 不做帧率限制、高精度 profiler
- 不重构 EngineLoop

## Decisions

### D1: 固定大小环形缓冲，窗口 60 帧

**选择**：`std::array<float, 60>` 作为环形缓冲，游标取模写入。

**替代方案**：EMA（指数移动平均）—— 无法精确控制"最近 N 帧"的窗口语义。

**理由**：简单、可预测、与 REQ 要求的"固定窗口滑动平均"完全匹配。60 帧 ≈ 1 秒 @60fps，是调试 FPS 显示的自然窗口。

### D2: 样本不足时按已有样本平均

**选择**：维护 `m_sampleCount`，`smoothedDeltaTime()` 除以 `min(m_sampleCount, 60)`。样本为零时回退到 `deltaTime()`。

**理由**：避免冷启动时除以零或显示不稳定值。

### D3: 测试用 sleep_for 做粗粒度验证

**选择**：`std::this_thread::sleep_for(1ms~5ms)` 制造可测量的时间差，验证非零性和单调性，不要求纳秒精度。

**理由**：跨平台可靠，测试重点是语义正确性而非精度。
