#pragma once
#include "core/input/input_state.hpp"

namespace LX_core {

class DummyInputState : public IInputState {
public:
  bool isKeyDown(KeyCode) const override { return false; }
  bool isMouseButtonDown(MouseButton) const override { return false; }
  Vec2f getMousePosition() const override { return {0, 0}; }
  Vec2f getMouseDelta() const override { return {0, 0}; }
  float getMouseWheelDelta() const override { return 0.0f; }
  void nextFrame() override {}
};

} // namespace LX_core
