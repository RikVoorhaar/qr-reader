---
name: doc-maintenance
description: Audit and repair documentation drift in AGENTS.md, README.md, and CONTEXT.md against the actual codebase. Use when the agent has made code changes that may invalidate docs, when the user asks "are the docs still accurate?", or when the AGENTS.md maintenance rule says "Run the doc-maintenance skill afterward to audit for drift."
---

# Doc Maintenance

## Quick start

```
Run in a fresh sub-agent:
  "Use the doc-maintenance skill to audit documentation against the codebase.
   I changed [describe what changed]. Mode: report (or: fix)"
```

## Two modes

- **report** — produce a drift report. Lists every inconsistency found. Does not edit anything.
- **fix** — produce a drift report AND apply the fixes. Only fix the items the drift report identifies; don't guess at additional changes.

## Audit checklist

For each, read the actual code and compare against the doc. Report every mismatch.

### 1. Module Map (AGENTS.md)

- Every `.py` file in `src/qr_reader/` must appear in the Module Map tables.
- Every file listed in the tables must exist on disk.
- The "Purpose" column must match the file's actual docstring / function.
- The "Depends on" column must list every internal `qr_reader` import in that file.

### 2. Data Flow (AGENTS.md)

- The function call sequence in the Data Flow diagram must match the actual call order in `detector/detector.py` `_run_detection()` and `decoder/decoder.py` `decode()`.
- Every function in the Data Flow must exist at the listed path.

### 3. Key Data Structures (AGENTS.md)

- Every data structure listed must have its fields match the actual `@dataclass` or `NamedTuple` definition.
- If a field was added or removed, flag it.

### 4. Common Modification Tasks (AGENTS.md)

- Every file path referenced in a task must exist.
- If a new module was added, consider whether a new task should be added (flag as suggestion, not error).

### 5. Architecture (README.md)

- The pipeline steps in README's "Architecture" section must match the Data Flow in AGENTS.md.
- The public API functions listed must match the actual public functions exposed.
- The module layout table must match the actual directories under `src/qr_reader/`.

### 6. CONTEXT.md

- Every term defined must still be used in the codebase (grep for the term).
- Terms missing: any domain term that appears in docstrings, class names, or function names but is NOT in CONTEXT.md (flag as suggestion).

### 7. Cross-document consistency

- README's Architecture and AGENTS.md's Data Flow must describe the same pipeline.
- Module names used across documents must be consistent.

## Output format

### If clean

```
✅ No drift detected. All documentation matches the codebase.
```

### If drift found

```md
## Drift Report

### Module Map
- `foo/bar.py` listed in table but file does not exist → remove from table
- `foo/baz.py` exists but is not in the table → add entry

### Data Flow
- `_run_detection` now calls `new_function()` between steps 5 and 6 → update diagram

### Key Data Structures
- `FinderPattern` gained field `confidence: float` → update definition

### Architecture (README.md)
- README lists 8 detection steps; code has 9 → update step count

### CONTEXT.md
- Term "FooBar" appears in `foo.py` docstring but is not in glossary → consider adding
```

## After fixing

If in fix mode, after editing the docs, re-run the audit in report mode to confirm zero drift. Do not end the session until the re-audit passes.
