---
name: openspec-e13-fix-change
description: Repair pending implementation findings from an OpenSpec change's verification report. Use after openspec-e13-verify-change reports failures. Does not revise planning artifacts or independently declare verification success.
license: MIT
---

Repair implementation defects recorded by `openspec-e13-verify-change`, using the current OpenSpec artifacts as the controlling scope.

**Repair boundary**: This workflow may edit implementation code and tests, and may update only the repair-owned fields in `<changeRoot>/verification.md`. Never revise proposal, specs, design, tasks, or task checkboxes. Never set the overall verification verdict or claim independent success.

**Store selection:** If the user names a store (a standalone OpenSpec repo registered on this machine) or the work lives in one, run `openspec store list --json`, select the store id, and pass `--store <id>` to OpenSpec commands that accept it. Keep that store selection for the whole workflow. Without a store, commands act on the nearest local OpenSpec root.

**Input**: Optionally specify a change name. If omitted, infer it from conversation context or auto-select the only active change. If multiple changes remain possible, run `openspec list --json` and ask the user to select one.

## Workflow

1. **Resolve the change and repair report**

   Announce `Using change: <name>` and how to specify another change.

   Run:

   ```bash
   openspec status --change "<name>" --json
   openspec instructions apply --change "<name>" --json
   ```

   Use the returned `schemaName`, `changeRoot`, `actionContext`, and `contextFiles`. Read `<changeRoot>/verification.md`. If it does not exist, does not identify the selected change, or contains no pending implementation findings, stop and explain what is missing; do not invent a repair list.

   An apply state of `all_done` is expected after initial implementation and does not stop this workflow. OpenSpec task completion and verification finding status are separate state machines. A blocked apply state caused by missing required artifacts does stop repair because the controlling scope is incomplete.

2. **Reload the controlling context**

   Read every file listed in `contextFiles` from disk, even if it was read earlier. Treat specs as the acceptance authority, proposal as scope, design as technical constraints, and project context as implementation constraints.

   The verification report narrows repair work but cannot override those artifacts. If repair guidance conflicts with current artifacts, or the artifacts changed enough to make the finding ambiguous, leave the finding pending and report the conflict to the planning or verification model.

3. **Select eligible findings**

   Work only on findings with:
   - `Category: implementation`; and
   - `Repair status: pending`.

   Never implement a `planning` or `environment` finding. Planning findings require `$openspec-update-change (Codex) or /openspec-update-change (other agents)` and a subsequent implementation decision. Environment findings require the stated environment condition to be resolved.

   Process independent eligible findings until all are addressed or a blocker prevents further safe progress.

4. **Reproduce before editing**

   For each finding, inspect its cited evidence and reproduce the issue when safe and practical. Confirm that the current implementation still exhibits the reported behavior.

   If it no longer reproduces, appears stale, lacks enough detail, or requires interpreting an unresolved product/design choice, do not guess and do not mark it addressed. Record the reason in `Repair notes` and continue only with findings that remain independent and unambiguous.

5. **Implement a bounded repair**

   Announce the finding ID being handled. Make the smallest coherent code and test changes that satisfy:
   - the cited requirement and scenario;
   - the finding's expected behavior and repair guidance;
   - relevant design and repository conventions.

   Add or strengthen a regression test when practical so the reported failure would have been caught. Do not broaden features, weaken acceptance criteria, remove failing coverage, or absorb unrelated cleanup.

6. **Check the repair**

   Run every applicable acceptance check listed in the finding, followed by focused regression checks for the affected area. Run broader tests, type checks, lint, or builds when the blast radius justifies them.

   Do not run destructive, production-facing, credential-dependent, or externally mutating checks without separate authorization. If required evidence cannot be obtained, leave the finding pending and report the blocker rather than treating code edits alone as success.

7. **Update only repair-owned report fields**

   After the repair and its available acceptance checks succeed, change only:

   ```text
   Repair status: addressed-awaiting-verification
   Repair notes: <files changed, tests added, commands run, and concise results>
   ```

   Preserve the finding ID, category, severity, requirement, evidence, expected behavior, actual behavior, repair guidance, acceptance checks, and reverification notes. Do not change the report's overall verdict. Only the independent verification workflow may mark `verified-resolved` or set `PASS`.

8. **Hand back for independent verification**

   Summarize:
   - findings addressed this session;
   - files changed;
   - checks run and results;
   - findings still pending or blocked and why.

   Always finish by directing the user back to `$openspec-e13-verify-change (Codex) or /openspec-e13-verify-change (other agents) <name>`. Repair completion is not verification completion.

## Guardrails

- Use `contextFiles` and `changeRoot` returned by the CLI; do not assume artifact or report paths relative to the current directory.
- Do not use `tasks.md` as the repair queue and do not alter its checkboxes.
- Do not rewrite verification findings to make the implementation conform on paper.
- Do not silently choose among contradictory or materially ambiguous requirements.
- Do not make speculative fixes for findings that cannot be reproduced or grounded in current artifacts.
- Keep unrelated user changes intact, including changes in files also touched by the repair.
- Continue through independent findings, but stop on any blocker that makes subsequent edits unsafe or scope-ambiguous.
- `addressed-awaiting-verification` means the repair model completed its work; it never means the finding is independently resolved.
