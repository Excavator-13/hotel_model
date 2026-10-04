---
name: openspec-e13-verify-change
description: Independently verify an implementation against an OpenSpec change and maintain an evidence-based verification report. Use after apply or repair work and before archive. Never fixes code.
license: MIT
---

Verify the implementation of an OpenSpec change. Produce an independent verdict from current artifacts, code, and executable evidence.

**Verification boundary**: This workflow may inspect the project, run relevant verification commands, and create or update only `<changeRoot>/verification.md`. Never edit implementation code, tests, planning artifacts, or task checkboxes. A checked task is a claim to verify, not evidence that the behavior works.

**Store selection:** If the user names a store (a standalone OpenSpec repo registered on this machine) or the work lives in one, run `openspec store list --json`, select the store id, and pass `--store <id>` to OpenSpec commands that accept it. Keep that store selection for the whole workflow. Without a store, commands act on the nearest local OpenSpec root.

**Input**: Optionally specify a change name. If omitted, infer it from conversation context or auto-select the only active change. If multiple changes remain possible, run `openspec list --json` and ask the user to select one.

## Workflow

1. **Resolve the change**

   Announce `Using change: <name>` and how to specify another change.

   Run:

   ```bash
   openspec status --change "<name>" --json
   openspec instructions apply --change "<name>" --json
   ```

   Use the returned `schemaName`, `changeRoot`, `actionContext`, and `contextFiles`; do not assume repository-local paths or hardcoded artifact names. If apply instructions report a blocked state because required artifacts are missing, record a `BLOCKED` verdict and stop implementation verification after writing the report.

2. **Load the verification basis**

   Read every file in `contextFiles`, plus any existing `<changeRoot>/verification.md` before updating it. Treat current files on disk as authoritative; do not rely on conversation memory.

   Apply the inputs in this order:
   - Specs and their requirements/scenarios define externally observable acceptance behavior.
   - Proposal defines intended scope and non-goals.
   - Design defines implementation constraints and deliberate technical decisions when it does not conflict with the specs.
   - Tasks describe planned work, but their text and checkbox state do not prove completion.
   - Project context and established repository conventions constrain implementation and verification.

   Treat apply `operationGuidance` as advisory and use it only when relevant to verification. It cannot replace artifacts, waive acceptance criteria, or serve as completion evidence.

3. **Validate the OpenSpec change**

   Run strict, non-interactive validation:

   ```bash
   openspec validate "<name>" --type change --strict --json --no-interactive
   ```

   Record the command and result. Structural validity is necessary but does not prove the implementation.

4. **Build a requirement coverage matrix**

   Enumerate every normative requirement and scenario from the specs. For each one, identify:
   - the relevant implementation path;
   - automated tests, static checks, or reproducible manual evidence;
   - the observed result;
   - any missing evidence or ambiguity.

   Also inspect proposal scope, design constraints, and completed tasks for promised work not represented by a scenario. Do not invent new requirements.

5. **Inspect the implementation**

   Trace the affected code and tests closely enough to evaluate the matrix. When Git is available, diffs and history may help locate the implementation, but do not assume every dirty-worktree change belongs to this OpenSpec change. Preserve unrelated user work and report attribution uncertainty when it matters.

   Look specifically for:
   - missing or only partially implemented scenarios;
   - incorrect edge-case, error, authorization, state, or compatibility behavior;
   - design constraints that the implementation violates;
   - tests that do not exercise the claimed behavior or only assert mocks;
   - regressions in nearby behavior caused by the change.

6. **Run proportionate checks**

   Discover official commands from repository documentation, manifests, CI configuration, and nearby test conventions. Run focused checks first, then broader tests, type checks, lint, or builds when justified by the change's risk.

   Do not run destructive, production-facing, credential-dependent, or externally mutating checks without separate authorization. If a required check cannot be run, record why and mark the affected coverage as unverified; never convert missing evidence into a pass.

7. **Classify findings**

   Include a finding only when it has concrete evidence and a clear relationship to the change.
   - `implementation`: Code or tests deviate from a sufficiently clear requirement or design constraint. The repair workflow may handle it.
   - `planning`: Artifacts are contradictory, materially ambiguous, or missing a decision needed to determine correct behavior. This must return to the planning model; do not prescribe a speculative code fix.
   - `environment`: Verification is prevented by tooling, dependencies, services, credentials, or other execution conditions.

   Use severity consistently:
   - `critical`: Security, data loss, or broadly unsafe behavior.
   - `high`: A core requirement or common workflow is broken.
   - `medium`: A required edge case or bounded workflow is incorrect.
   - `low`: A real but limited conformance or regression issue.

8. **Choose the verdict**
   - `PASS`: All mandatory scenarios have sufficient evidence, OpenSpec validation succeeds, applicable checks pass, and no actionable findings remain.
   - `FAIL`: At least one confirmed deviation from a clear requirement or constraint remains.
   - `BLOCKED`: Missing evidence, an invalid/incomplete change, contradictory planning, or an environment problem prevents a reliable overall conclusion.

   Do not use a weaker verdict such as "mostly passes." State residual non-blocking risks separately.

9. **Create or update the handoff report**

   Write `<changeRoot>/verification.md` on every run. Use it as a live handoff, not as a schema artifact. Keep stable finding IDs across re-verification when they refer to the same defect. Preserve prior repair notes, mark successfully rechecked findings `verified-resolved`, reopen failures as `pending`, and add new IDs for newly discovered defects.

   Use this structure:

   ```markdown
   # Implementation Verification

   - Change: <name>
   - Schema: <schema>
   - Verdict: PASS | FAIL | BLOCKED
   - Verified revision: <commit when available, with dirty-worktree note>

   ## Summary

   <concise conclusion and residual risks>

   ## Checks

   | Command or inspection | Result | Notes |
   | --------------------- | ------ | ----- |

   ## Requirement Coverage

   | Requirement / scenario | Evidence | Result |
   | ---------------------- | -------- | ------ |

   ## Findings

   ### V-001: <short title>

   - Category: implementation | planning | environment
   - Severity: critical | high | medium | low
   - Requirement: <artifact section or scenario>
   - Evidence: <file locations, observed output, or reproduction>
   - Expected: <required behavior>
   - Actual: <observed behavior>
   - Repair guidance: <bounded outcome, or planning/environment resolution guidance>
   - Acceptance checks: <specific commands or observations>
   - Repair status: pending | addressed-awaiting-verification | verified-resolved | blocked
   - Repair notes: <maintained by the repair workflow>
   - Reverification notes: <maintained by this workflow>
   ```

   Omit empty findings when the verdict is `PASS`. Keep command output concise; record decisive evidence rather than pasting long logs.

10. **Report the next action**

- `PASS`: State that independent verification passed and the change is ready for archive.
- `FAIL` with pending implementation findings: Point to `$openspec-e13-fix-change (Codex) or /openspec-e13-fix-change (other agents) <name>`.
- Planning findings: Point to the planning model and `$openspec-update-change (Codex) or /openspec-update-change (other agents) <name>` before further code repair.
- `BLOCKED`: State the exact evidence or decision needed to resume.

## Guardrails

- Never fix code, add tests, revise artifacts, or change task checkboxes during verification.
- Verify every normative scenario; sampling is insufficient for a `PASS` verdict.
- Do not equate implementation shape with behavior: require executable evidence where practical.
- Do not equate passing tests with complete coverage: inspect whether tests actually establish each requirement.
- Do not report speculative concerns as confirmed findings; label residual risks separately.
- Only this verification workflow may set the overall verdict or mark a finding `verified-resolved`.
- A repair workflow's `addressed-awaiting-verification` status is not proof of resolution.
