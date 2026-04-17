## ADDED Requirements

### Requirement: debug_ui module location and namespace

A new helper module SHALL be created at `src/infra/gui/debug_ui.hpp` and `src/infra/gui/debug_ui.cpp`, using the namespace `LX_infra::debug_ui`. The module SHALL live inside the `LX_Infra` library (registered via `src/infra/CMakeLists.txt`'s `INFRA_SOURCES`) and SHALL NOT be included from any file under `src/core/`. ImGui headers SHALL only be included inside the `.cpp`; the public `debug_ui.hpp` SHALL NOT expose any ImGui type.

#### Scenario: debug_ui does not leak ImGui into core

- **WHEN** compiling any translation unit under `src/core/`
- **THEN** no `#include "infra/gui/debug_ui.hpp"` SHALL be required to link, and no ImGui header SHALL reach the core translation units through debug_ui

#### Scenario: debug_ui.hpp has no ImGui types in its signatures

- **WHEN** grepping `src/infra/gui/debug_ui.hpp` for `ImGui` / `ImVec` / `ImDrawData`
- **THEN** no match SHALL appear outside of comments

### Requirement: Vector and scalar bridging helpers

`LX_infra::debug_ui` SHALL provide the following free functions with ImGui widget behavior:

```cpp
bool dragVec3(const char* label, LX_core::Vec3f& value,
              float speed = 0.01f, float min = 0.0f, float max = 0.0f);
bool dragVec4(const char* label, LX_core::Vec4f& value,
              float speed = 0.01f, float min = 0.0f, float max = 0.0f);
bool sliderFloat(const char* label, float& value, float min, float max);
bool sliderInt(const char* label, int& value, int min, int max);
bool colorEdit3(const char* label, LX_core::Vec3f& rgb);
bool colorEdit4(const char* label, LX_core::Vec4f& rgba);
```

Each helper SHALL return `true` when the widget reports a user-driven change this frame, matching ImGui's own return-value convention. The `Vec3f` / `Vec4f` helpers SHALL bridge to ImGui `float*` widgets without additional allocation; the implementation file SHALL include `static_assert`s that `sizeof(Vec3f) == 3 * sizeof(float)` and `sizeof(Vec4f) == 4 * sizeof(float)` to catch future math-type layout drift at compile time.

#### Scenario: dragVec3 reports change

- **WHEN** `dragVec3("pos", value)` is called during an ImGui frame and the user drags the widget
- **THEN** `value.x/y/z` SHALL reflect the new values and the call SHALL return `true`

#### Scenario: Vec layout drift is caught at compile time

- **WHEN** `Vec3f` or `Vec4f` layout changes such that `sizeof(VecN) != N * sizeof(float)`
- **THEN** compilation of `debug_ui.cpp` SHALL fail with a static assertion error

### Requirement: Label and StringID display helpers

`LX_infra::debug_ui` SHALL provide:

```cpp
void labelText(const char* label, const char* value);
void labelText(const char* label, const std::string& value);
void labelFloat(const char* label, float value);
void labelInt(const char* label, int value);
void labelStringId(const char* label, LX_core::StringID value);
```

`labelStringId` SHALL resolve the displayed text via `LX_core::GlobalStringTable::get().getName(value.id)`. When the returned name is empty, the helper SHALL display a non-empty placeholder (e.g. `(empty)` or `#<id>`) so that a missing registration never produces a visually empty row.

#### Scenario: labelStringId renders the registered name

- **WHEN** `labelStringId("pass", StringID("forward"))` is called and "forward" has been registered in `GlobalStringTable`
- **THEN** the rendered text for the value column SHALL equal "forward"

#### Scenario: labelStringId falls back when name is empty

- **WHEN** `labelStringId("pass", StringID{})` is called with an id whose name resolves to an empty string
- **THEN** the rendered text SHALL NOT be empty; it SHALL be a non-empty placeholder the user can see

### Requirement: Panel and section container helpers

`LX_infra::debug_ui` SHALL provide:

```cpp
bool beginPanel(const char* title);
void endPanel();
bool beginSection(const char* title);
void endSection();
void separatorText(const char* label);
```

Behavior:

- `beginPanel` SHALL apply a consistent default position and size policy (for example, `ImGuiCond_FirstUseEver` position `{8, 8}` and size `{320, 400}`) and call `ImGui::Begin(title)`; it SHALL return ImGui's `Begin` return value
- `endPanel` SHALL unconditionally call `ImGui::End()`, matching ImGui's rule that `Begin` must always be paired with `End` regardless of `Begin`'s return value
- `beginSection` SHALL render a collapsible group (e.g. `CollapsingHeader`) and return `true` when its body SHALL be drawn
- `endSection` SHALL be safe to call after any `beginSection`, even when the underlying primitive does not require an explicit close call (it MAY be a no-op)
- `separatorText` SHALL render a labelled separator

#### Scenario: beginPanel/endPanel are always paired

- **WHEN** `beginPanel("X")` returns `false`
- **THEN** the caller SHALL still call `endPanel()` without error or assertion

#### Scenario: beginSection/endSection are always paired

- **WHEN** `beginSection("Advanced")` returns `false` because the section is collapsed
- **THEN** the caller SHALL still call `endSection()` and the call SHALL succeed

### Requirement: renderStatsPanel shows frame timing

`LX_infra::debug_ui` SHALL provide:

```cpp
void renderStatsPanel(const LX_core::Clock& clock);
```

The panel SHALL, at minimum, display:

1. The current frame count, derived from `clock.frameCount()`
2. The current frame's delta time in milliseconds, derived from `clock.deltaTime()`
3. A smoothed FPS value, derived from `clock.smoothedDeltaTime()` (FPS = `1.0 / smoothedDeltaTime` when non-zero; a sentinel value otherwise)

The helper SHALL NOT mutate the clock.

#### Scenario: renderStatsPanel renders without mutating clock

- **WHEN** `renderStatsPanel(clock)` is called inside an active ImGui frame
- **THEN** frame count, delta time (ms), and a smoothed FPS value SHALL be rendered; the clock's internal state SHALL remain unchanged

### Requirement: cameraPanel edits camera fields without side effects

`LX_infra::debug_ui` SHALL provide:

```cpp
void cameraPanel(const char* title, LX_core::Camera& camera);
```

The panel SHALL allow viewing and editing at least these fields:

- `camera.position` (Vec3f)
- `camera.target` (Vec3f)
- `camera.up` (Vec3f)
- `camera.fovY` (float)
- `camera.aspect` (float)
- `camera.nearPlane` (float)
- `camera.farPlane` (float)

The helper SHALL NOT call `camera.updateMatrices()` internally. The caller is responsible for deciding when to refresh matrices.

#### Scenario: cameraPanel edits position without calling updateMatrices

- **WHEN** the user drags the position widget so that `camera.position` changes from `{0,0,0}` to `{1,2,3}`
- **THEN** `camera.position` SHALL equal `{1,2,3}` after the call AND `camera` view/projection matrices SHALL NOT have been automatically refreshed by the helper

### Requirement: directionalLightPanel edits UBO params and marks dirty on change

`LX_infra::debug_ui` SHALL provide:

```cpp
void directionalLightPanel(const char* title, LX_core::DirectionalLight& light);
```

The panel SHALL work directly against the real `light.ubo->param` layout (which currently consists of `Vec4f dir` and `Vec4f color`), and SHALL, at minimum:

1. Allow editing `light.ubo->param.dir`
2. Allow editing `light.ubo->param.color`
3. Call `light.ubo->setDirty()` when any edit is made during that frame

The helper SHALL NOT invent a higher-level lighting model (e.g. separate `direction`, `intensity`, `color` fields) that does not match the current `DirectionalLight` data layout.

#### Scenario: directionalLightPanel marks dirty on edit

- **WHEN** the user edits the direction or color widget such that `light.ubo->param.dir` or `light.ubo->param.color` changes this frame
- **THEN** `light.ubo->isDirty()` SHALL return `true` after the helper returns

#### Scenario: directionalLightPanel does not mark dirty when untouched

- **WHEN** `directionalLightPanel(...)` is called in a frame during which the user does not interact with any widget (all widgets report `changed == false`)
- **THEN** the helper SHALL NOT call `setDirty()` for this reason (pre-existing dirty state is unaffected)

### Requirement: debug_ui explicitly excludes material reflection editor

`LX_infra::debug_ui` SHALL NOT provide a `materialPanel()` or any reflection-driven material editor in this change. Callers that need material-specific UI SHALL write bespoke ImGui code against `MaterialInstance` / `ShaderResourceBinding` directly for now.

#### Scenario: No materialPanel symbol is exported

- **WHEN** grepping `src/infra/gui/debug_ui.hpp` for `materialPanel`
- **THEN** no match SHALL appear

### Requirement: Integration smoke test

`src/test/integration/test_debug_ui_smoke.cpp` SHALL be added and registered in `src/test/CMakeLists.txt`. The test SHALL verify:

1. Every public helper symbol declared in `debug_ui.hpp` resolves at link time (e.g. by capturing function-pointer addresses into a `void*` array and asserting all entries are non-null)
2. When ImGui context creation is possible in the test environment (which only requires a CPU-side `ImGui::CreateContext`, no window / no GPU), a minimal smoke sequence SHALL run: create context → `NewFrame` → invoke the main helper surface (at least `beginPanel` / `dragVec3` / `colorEdit4` / `labelStringId` / one composite panel) → `EndFrame` → `DestroyContext`, with no assertions firing

Pixel-level or screenshot verification is out of scope. When ImGui context creation is not available, link-level verification alone SHALL be sufficient to pass the test.

#### Scenario: Link-level symbols are reachable

- **WHEN** running `test_debug_ui_smoke`
- **THEN** the function-pointer table SHALL contain only non-null entries

#### Scenario: CPU-only ImGui smoke runs cleanly

- **WHEN** running `test_debug_ui_smoke` in an environment that allows `ImGui::CreateContext()` (no GPU / display required)
- **THEN** the minimal `NewFrame → helpers → EndFrame` sequence SHALL complete without crash or assertion
