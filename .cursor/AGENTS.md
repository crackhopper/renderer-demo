# Agent Guidance

## Shell Environment

This project runs on **Windows with PowerShell**. The AI assistant SHALL default to PowerShell syntax for all shell commands.

### Key Rules

1. **Default to PowerShell**: All shell commands MUST use PowerShell cmdlets and syntax
2. **Windows paths**: Use `C:\Users\...` format, not Unix paths
3. **PowerShell cmdlets**:
   - `Get-ChildItem` or `dir` for listing
   - `Copy-Item`, `Move-Item`, `Remove-Item` for file operations
   - `-Filter` for glob patterns
4. **Output suppression**: Use `$null` instead of `/dev/null`
5. **Command chaining**: Use `;` or `&` instead of `&&` or `||`

### Quick Reference

```powershell
# List files
Get-ChildItem -Force

# Copy files
Copy-Item -Path "src.txt" -Destination "dst.txt"

# Find files recursively
Get-ChildItem -Recurse -Filter "*.txt"

# Silent execution
cmd > $null 2>&1

# Chain commands
cmd1; cmd2
```

### When to Use Other Shells

Only deviate from PowerShell when:
- User explicitly requests bash or another shell
- Target is WSL/Linux environment
- A specific tool only supports bash syntax

## C++ Coding Standards

This project follows strict C++ coding standards defined in `openspec/specs/cpp-style-guide/spec.md`. **CRITICAL**: AI assistants MUST read and follow these guidelines.

### Key Policies

1. **No Raw Pointers for Object References**: Raw pointers (`T*`) SHALL NOT be used to hold references to other objects. Use `std::unique_ptr<T>`, `std::shared_ptr<T>`, or references (`T&`) instead.

2. **Constructor Injection Only**: All dependencies MUST be injected via constructor. No setter methods for dependency injection.

3. **GPU Object Return Types**: GPU objects (Vulkan wrappers, etc.) MUST be returned via `std::unique_ptr<T>` from factory functions, NOT by value.

4. **CommandBuffer Design**: `VulkanCommandBuffer` MUST NOT hold references to manager objects. Pass manager references as operation parameters.

5. **Ownership Clarity**: Function parameters and return types must have explicit ownership semantics.

### When Working on C++ Code

1. **Before making changes**: Read `openspec/specs/cpp-style-guide/spec.md`
2. **Follow the conventions**: Use `std::unique_ptr` for owned objects, references for non-owning dependencies
3. **Update the spec**: If you discover a new pattern that should be standardized, propose an update to the style guide

See `openspec/specs/cpp-style-guide/spec.md` for full details including:
- Complete ownership guidelines
- Exception cases (handle pointers, C interop)
- Code examples (correct vs incorrect patterns)
- Vulkan backend-specific conventions
