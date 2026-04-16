## MODIFIED Requirements

### Requirement: Model file path convention
Model test assets SHALL be located under `assets/models/` instead of the root-level `models/` directory. Any code or documentation referencing `models/viking_room.obj` SHALL be updated to reference `assets/models/viking_room/viking_room.obj`.

#### Scenario: Viking room model at new path
- **WHEN** loading the viking_room model
- **THEN** the file path SHALL resolve to `assets/models/viking_room/viking_room.obj`

#### Scenario: Old models directory does not exist
- **WHEN** checking the repository root
- **THEN** no `models/` directory SHALL exist at the root level
