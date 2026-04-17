## ADDED Requirements

### Requirement: Window exposes native handle

The `Window` interface (`src/core/platform/window.hpp`) SHALL declare a pure virtual method:

```cpp
virtual void* getNativeHandle() const = 0;
```

The SDL implementation (`src/infra/window/sdl_window.cpp`) SHALL return the underlying `SDL_Window*`. The GLFW implementation (`src/infra/window/glfw_window.cpp`) SHALL return the underlying `GLFWwindow*`. The `core` layer SHALL NOT include any SDL or GLFW headers; the conversion from `void*` back to the concrete type SHALL happen only in backend/infra code that already depends on the corresponding library.

#### Scenario: SDL window returns SDL_Window pointer

- **WHEN** `getNativeHandle()` is called on an SDL-backed `Window`
- **THEN** the returned pointer SHALL be the same address returned by the internal `SDL_CreateWindow` call

#### Scenario: GLFW window returns GLFWwindow pointer

- **WHEN** `getNativeHandle()` is called on a GLFW-backed `Window`
- **THEN** the returned pointer SHALL be the same address returned by the internal `glfwCreateWindow` call

#### Scenario: core layer does not include SDL or GLFW headers

- **WHEN** compiling a translation unit that includes only `core/platform/window.hpp`
- **THEN** compilation SHALL succeed without SDL or GLFW headers being available on the include path

## MODIFIED Requirements

### Requirement: Window close detection

`shouldClose()` SHALL poll all SDL events in a single loop. For every event, it SHALL, in order:

1. Call `ImGui_ImplSDL3_ProcessEvent(&event)` so that ImGui can observe focus, input, and resize events. The SDL backend implementation SHALL guarantee this call is a no-op when ImGui has not been initialized yet (either by guarding with its own initialization flag or by relying on `ImGui_ImplSDL3_ProcessEvent`'s documented early-return behavior).
2. Forward the event to `Sdl3InputState::handleSdlEvent()` so the input state reflects the latest key/mouse data.
3. Check for `SDL_EVENT_QUIT` and remember whether quit was seen.

Events SHALL NOT be consumed twice and SHALL NOT be re-polled in a second loop. After the loop, `shouldClose()` SHALL return `true` if any `SDL_EVENT_QUIT` was seen in this invocation.

#### Scenario: shouldClose forwards events to ImGui, input state, and detects quit

- **WHEN** `shouldClose()` is called and SDL events include a key press and a quit event
- **THEN** `ImGui_ImplSDL3_ProcessEvent` SHALL have been invoked for each event, the key press state SHALL be updated in `Sdl3InputState`, and `shouldClose()` SHALL return `true`

#### Scenario: shouldClose updates input without quit

- **WHEN** `shouldClose()` is called and SDL events include only key presses and mouse motion
- **THEN** all events SHALL be forwarded to both ImGui and `Sdl3InputState`, and `shouldClose()` SHALL return `false`
