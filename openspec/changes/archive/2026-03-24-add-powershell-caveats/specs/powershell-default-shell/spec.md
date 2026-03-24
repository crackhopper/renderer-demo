## ADDED Requirements

### Requirement: PowerShell as Default Shell
The AI assistant SHALL default to PowerShell syntax for all shell commands when operating in a Windows/PowerShell environment.

#### Scenario: Generate directory listing
- **WHEN** the user asks to list files in a directory
- **THEN** the AI SHALL use `Get-ChildItem` or `dir` instead of `ls`

#### Scenario: Generate path operations
- **WHEN** the user asks to work with file paths
- **THEN** the AI SHALL use Windows path conventions (e.g., `C:\Users\...`) instead of Unix paths

#### Scenario: Generate command chaining
- **WHEN** the user asks to run multiple commands sequentially
- **THEN** the AI SHALL use PowerShell's `;` or `&` operators appropriately

### Requirement: PowerShell Glob Patterns
The AI assistant SHALL use PowerShell-compatible glob patterns for file operations.

#### Scenario: Generate file filtering
- **WHEN** the user asks to find files matching a pattern
- **THEN** the AI SHALL use `-Filter` parameter with `Get-ChildItem` instead of bash glob patterns like `*.txt`

#### Scenario: Generate recursive file search
- **WHEN** the user asks to recursively search for files
- **THEN** the AI SHALL use `Get-ChildItem -Recurse` instead of `find` or `ls -R`
