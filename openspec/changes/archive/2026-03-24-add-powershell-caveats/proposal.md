## Why

Currently, the AI assistant defaults to bash syntax when generating shell commands, but the development environment runs on Windows with PowerShell as the shell. This mismatch causes users to receive commands that require translation or fail outright when executed directly.

## What Changes

- Add system prompt guidance instructing the AI to always use PowerShell syntax for shell commands
- Include explicit PowerShell syntax examples in the system prompt for common operations
- Flag bash-specific patterns that require conversion to PowerShell equivalents

## Capabilities

### New Capabilities

- `powershell-default-shell`: System prompt configuration that establishes PowerShell as the default shell environment, ensuring all shell commands, glob patterns, and path conventions default to PowerShell syntax rather than bash

### Modified Capabilities

- (none)

## Impact

- AI assistant behavior when generating shell commands
- All `/opsx-*` commands and workflow automation that invoke shell operations
