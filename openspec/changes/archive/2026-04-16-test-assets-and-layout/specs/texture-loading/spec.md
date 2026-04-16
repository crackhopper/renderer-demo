## MODIFIED Requirements

### Requirement: Texture file path convention
Texture test assets SHALL be located under `assets/textures/` instead of the root-level `textures/` directory. Any code or documentation referencing `textures/viking_room.png` SHALL be updated to reference `assets/textures/viking_room/viking_room.png`.

#### Scenario: Viking room texture at new path
- **WHEN** loading the viking_room texture
- **THEN** the file path SHALL resolve to `assets/textures/viking_room/viking_room.png`

#### Scenario: Old textures directory does not exist
- **WHEN** checking the repository root
- **THEN** no `textures/` directory SHALL exist at the root level
