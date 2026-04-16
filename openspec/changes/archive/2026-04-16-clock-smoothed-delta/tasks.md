## 1. Clock 平滑 deltaTime 实现

- [x] 1.1 在 `clock.hpp` 新增 `smoothedDeltaTime()` 声明和内部状态（环形缓冲 array<float,60>、游标、样本数）
- [x] 1.2 在 `clock.cpp` 的 `tick()` 中，从第二次 tick 开始将 deltaTime 写入环形缓冲
- [x] 1.3 在 `clock.cpp` 实现 `smoothedDeltaTime()`（样本为零回退到 deltaTime，否则按已有样本平均）

## 2. 集成测试

- [x] 2.1 创建 `src/test/integration/test_clock.cpp`，覆盖 5 个测试场景
- [x] 2.2 在 `src/test/CMakeLists.txt` 注册 `test_clock`
- [x] 2.3 构建并运行测试验证通过

## 3. 收尾

- [x] 3.1 更新 REQ-014 实施状态
