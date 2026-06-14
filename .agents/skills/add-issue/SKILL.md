---
name: add-issue
description: Create a new issue in the project's lightweight issue tracker. Use when the agent or maintainer wants to file a bug, feature request, enhancement, docs task, or chore.
---

# Add Issue

Create a new issue file in `issues/` and update the index.

## Before creating

- Check `issues/README.md` for duplicates.
- For speculative enhancements, confirm with the maintainer first.
- For bugs you can fix inline in the current task, fix them — don't file an issue.

## Steps

### 1. Determine the next issue number

Look at the Open and Closed tables in `issues/README.md`. The next number is
one higher than the highest existing number. If no issues exist, start at 1.
Numbers are zero-padded to 4 digits (`0001`, `0002`, etc.).

### 2. Choose a slug

Derive a short, hyphenated slug from the title. Examples:
- `fix-version-estimation-seed-42`
- `add-kanji-mode`
- `improve-finder-pattern-detection`

Keep it concise — 5–8 words max.

### 3. Write the issue file

Create `issues/NNNN-slug.md` with this frontmatter block:

```yaml
---
title: <short description>
tags: [bug | feature | enhancement | docs | chore]
priority: blocking | high | medium | low
status: open
created: YYYY-MM-DD
---

## Description

...
```

- `created` is today's date.
- Do not include `closed` — it only appears when an issue is closed.
- If you don't have enough detail for a full description, write a placeholder
  sentence and add `[NEEDS DETAIL]` so the maintainer knows to expand it.

### 4. Add the row to `issues/README.md`

Insert a row into the **Open** table:

```markdown
| [NNNN](./NNNN-slug.md) | Title | category | priority | open |
```

Sort the table: `blocking` first, then `high`, `medium`, `low`. Within the same
priority, newer issues go below older ones (ascending by number).

Remove the `_No open issues._` placeholder if it's there.

### 5. Tell the user

Output: `Created [issues/NNNN-slug.md](./issues/NNNN-slug.md).`
