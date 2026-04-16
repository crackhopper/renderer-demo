## MODIFIED Requirements

### Requirement: Window provides input state access
The SDL `Window` implementation SHALL return a `Sdl3InputState` (not `DummyInputState`) from `getInputState()`. The GLFW implementation SHALL continue returning `DummyInputState`.

#### Scenario: SDL window returns real input state
- **WHEN** calling `getInputState()` on the SDL window implementation
- **THEN** the returned object SHALL be a `Sdl3InputState` that responds to SDL events

#### Scenario: GLFW window still returns DummyInputState
- **WHEN** calling `getInputState()` on the GLFW window implementation
- **THEN** the returned object SHALL be a `DummyInputState` with all-zero semantics

### Requirement: Window close detection
`shouldClose()` SHALL poll all SDL events in a single loop, forwarding each event to `Sdl3InputState::handleSdlEvent()` before checking for quit. Events SHALL NOT be consumed twice.

#### Scenario: shouldClose updates input and detects quit in one pass
- **WHEN** `shouldClose()` is called and SDL events include both key presses and a quit event
- **THEN** key press states SHALL be updated in the input state AND `shouldClose()` SHALL return `true`

#### Scenario: shouldClose updates input without quit
- **WHEN** `shouldClose()` is called and SDL events include only key presses
- **THEN** key press states SHALL be updated and `shouldClose()` SHALL return `false`
