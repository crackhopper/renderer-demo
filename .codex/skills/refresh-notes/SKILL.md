---
name: refresh-notes
description: Restart the local notes site after docs changes by running the project refresh script and confirming the mkdocs serve process came back up. Use after updating notes, requirements, or mkdocs inputs.
---

Refresh the local notes site after documentation changes.

## Workflow

1. Run:

```bash
scripts/refresh-notes.sh
```

2. Confirm that:
   - `mkdocs.gen.yml` was regenerated
   - any old listener on the notes port was stopped
   - a fresh `mkdocs serve` process started in the background
   - the script reports URL, PID, and log path
3. Report a concise summary and any missing prerequisite such as `python3`.

## Guardrails

- If restart fails, report the exact failing step.
