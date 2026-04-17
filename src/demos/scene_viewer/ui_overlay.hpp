#pragma once

// REQ-019: ImGui overlay for the scene viewer demo. Binds to scene objects
// via raw non-owning pointers; lifetime is the lifetime of main(). Hotkey
// edge detection (F1 toggles help) is done locally.

#include "core/input/input_state.hpp"
#include "core/scene/camera.hpp"
#include "core/scene/light.hpp"
#include "core/time/clock.hpp"

namespace LX_demo::scene_viewer {

class CameraRig;

class UiOverlay {
public:
  void attach(const LX_core::Clock* clock,
              LX_core::Camera* camera,
              LX_core::DirectionalLight* light,
              CameraRig* rig);

  // Called from the VulkanRenderer UI callback each frame.
  void drawFrame();

  // Called from the EngineLoop update hook before the rig update.
  void handleHotkeys(LX_core::IInputState& input);

private:
  const LX_core::Clock* m_clock = nullptr;
  LX_core::Camera* m_camera = nullptr;
  LX_core::DirectionalLight* m_light = nullptr;
  CameraRig* m_rig = nullptr;
  bool m_prevF1Down = false;
  bool m_helpVisible = true;
};

} // namespace LX_demo::scene_viewer
