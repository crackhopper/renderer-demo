## ADDED Requirements

### Requirement: Sdl3InputState implements IInputState
`Sdl3InputState` in `src/infra/window/sdl3_input_state.hpp/.cpp` SHALL implement `LX_core::IInputState` with SDL3 event-driven state updates.

#### Scenario: Key down and up tracking
- **WHEN** `handleSdlEvent()` receives `SDL_EVENT_KEY_DOWN` for key W
- **THEN** `isKeyDown(KeyCode::W)` SHALL return `true`
- **WHEN** `handleSdlEvent()` subsequently receives `SDL_EVENT_KEY_UP` for key W
- **THEN** `isKeyDown(KeyCode::W)` SHALL return `false`

#### Scenario: Mouse button down and up tracking
- **WHEN** `handleSdlEvent()` receives `SDL_EVENT_MOUSE_BUTTON_DOWN` for left button
- **THEN** `isMouseButtonDown(MouseButton::Left)` SHALL return `true`
- **WHEN** `handleSdlEvent()` subsequently receives `SDL_EVENT_MOUSE_BUTTON_UP` for left button
- **THEN** `isMouseButtonDown(MouseButton::Left)` SHALL return `false`

#### Scenario: Mouse position updates
- **WHEN** `handleSdlEvent()` receives `SDL_EVENT_MOUSE_MOTION` with x=100, y=200
- **THEN** `getMousePosition()` SHALL return `{100, 200}`

#### Scenario: Mouse delta accumulates within frame
- **WHEN** `handleSdlEvent()` receives two motion events with xrel=5,yrel=3 and xrel=2,yrel=1
- **THEN** `getMouseDelta()` SHALL return `{7, 4}`

#### Scenario: Mouse wheel delta accumulates within frame
- **WHEN** `handleSdlEvent()` receives two wheel events with y=1.0 and y=0.5
- **THEN** `getMouseWheelDelta()` SHALL return `1.5`

#### Scenario: nextFrame clears delta but preserves down state
- **WHEN** `nextFrame()` is called after accumulating mouse delta and wheel delta while a key is held down
- **THEN** `getMouseDelta()` SHALL return `{0, 0}`, `getMouseWheelDelta()` SHALL return `0.0f`, and `isKeyDown()` for the held key SHALL still return `true`

#### Scenario: handleSdlEvent returns true on quit
- **WHEN** `handleSdlEvent()` receives `SDL_EVENT_QUIT`
- **THEN** the return value SHALL be `true`

#### Scenario: handleSdlEvent returns false on non-quit
- **WHEN** `handleSdlEvent()` receives any non-quit event
- **THEN** the return value SHALL be `false`

### Requirement: SDL scancode to KeyCode mapping
A mapping function SHALL convert `SDL_Scancode` to `LX_core::KeyCode`, covering all keys defined in REQ-012: A-Z, Num0-Num9, Escape, Space, LShift, RShift, LCtrl, RCtrl, LAlt, RAlt, Enter, Tab, arrow keys, F1-F4. Unmapped scancodes SHALL map to `KeyCode::Unknown`.

#### Scenario: Letter key mapping
- **WHEN** an SDL event with scancode `SDL_SCANCODE_W` is received
- **THEN** it SHALL map to `KeyCode::W`

#### Scenario: Unknown scancode mapping
- **WHEN** an SDL event with an unmapped scancode (e.g., F12) is received
- **THEN** it SHALL map to `KeyCode::Unknown` and the event SHALL be ignored

### Requirement: Integration test for Sdl3InputState
`src/test/integration/test_sdl_input.cpp` SHALL test `Sdl3InputState` by manually constructing `SDL_Event` structs and calling `handleSdlEvent()`.

#### Scenario: All SDL input tests pass
- **WHEN** running `test_sdl_input`
- **THEN** all assertions SHALL pass
