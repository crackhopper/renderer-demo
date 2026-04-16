## 1. FreeFlyCameraController 实现

- [x] 1.1 Create `src/core/scene/freefly_camera_controller.hpp` — class declaration with position, yaw, pitch state and tuning fields
- [x] 1.2 Create `src/core/scene/freefly_camera_controller.cpp` — implement update(): right-button mouse look, WASD+Space/LShift movement, diagonal normalization, LCtrl boost, camera writeback
- [x] 1.3 Verify `src/core/CMakeLists.txt` GLOB_RECURSE covers new .cpp (no change needed)

## 2. 集成测试

- [x] 2.1 Create `src/test/integration/test_freefly_camera_controller.cpp` — tests: w_key_moves_forward, mouse_look_only_with_right_button, diagonal_normalization, boost_multiplies_speed, pitch_clamped
- [x] 2.2 Update `src/test/CMakeLists.txt` — register `test_freefly_camera_controller` in TEST_INTEGRATION_EXE_LIST

## 3. 验证

- [x] 3.1 Build passes (LX_Core and test target)
- [x] 3.2 All freefly controller tests pass
