---
name: sync-design-docs
description: Sync AGENTS.md and CLAUDE.md design-doc indexes with notes/subsystems/. Use when subsystem docs changed and the top-level design indexes need to point at the current notes/subsystems set.
---

Synchronize the design-doc indexes in `AGENTS.md` and `CLAUDE.md` with `notes/subsystems/`.

## Core Rules

- Treat `notes/subsystems/` as the source of truth.
- Keep summaries concise.
- Use English summaries in `AGENTS.md` and `CLAUDE.md`.
- Remove stale index entries instead of preserving dead paths.

## Workflow

1. Read the design-doc index sections in:
   - `AGENTS.md`
   - `CLAUDE.md`
2. List all `*.md` files under `notes/subsystems/`.
3. Diff the indexes against the actual subsystem-doc set.
4. For each missing or changed subsystem doc:
   - read the doc
   - generate a short summary
   - update both indexes
5. Remove stale paths that no longer exist.
6. If `.cursorrules` has a design-doc section, sync that too.
7. Report what was added, updated, removed, or already in sync.
