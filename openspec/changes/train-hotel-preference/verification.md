# Implementation Verification

- Change: train-hotel-preference
- Schema: spec-driven
- Verdict: FAIL
- Verified revision: `e34dca5351c6c41b8e02ede8438ed360dbf83898` (`apply train-hotel-preference`). Worktree was clean before verification; this report is the only verification edit.

## Summary

Strict OpenSpec validation succeeds and the independently executed full suite reports **228 passed**. However, the answer-first checkpoint is not recoverable if execution stops before the second model checkpoint: it contains an additional label without updating the model's lagging-label count. Real SIGKILL, an interrupted fit, and a model-save fsync failure reproduce this defect. Three limited conformance issues also remain. The change is not ready for archive.

All eight requirements and all twenty named scenarios were inspected. Passing existing tests are not sufficient evidence for the two-stage checkpoint guarantee. In particular, `tests/test_state.py:238` (`test_answer_persisted_before_fitting`) only creates and loads an empty session; it never saves an answer, interrupts fitting, or checks recovery of the intermediate checkpoint. Existing optimizer-failure integration tests allow the failure handler to write another checkpoint, concealing the intermediate-state defect. Task 4.2's checked box therefore does not establish its claimed interruption coverage.

The three previously reported fixes were independently rechecked: skip-only sessions remain untrained without a fit-failure message; trained `ModelSnapshot` equality and serialized equality work; all three README training commands execute successfully. The CLI help still has a separate command-documentation defect (V-002).

Residual observations and evidence limits:

- Real SIGINT was delivered to a CLI subprocess blocked on pipe input, and exit/recovery succeeded. A physical Ctrl-C keystroke through a real TTY was not exercised. The existing injected `KeyboardInterrupt` test and real signal evidence cover the input-interruption behavior, but do not establish terminal-driver behavior.
- Real SIGKILL was exercised after the answer checkpoint and before fitting, not during a kernel rename or disk flush. Fsync failure was injected at the file-operation boundary. Physical disk faults, power loss, and directory-entry durability were not tested; those stronger guarantees are not promised by the artifacts.
- Fingerprint sorting and omission of CSV row order, BOM, and unrelated columns match design decision 6. Independent checks also showed that changing any of the eleven required fields changes the fingerprint.
- Default exploration has a direct `rng.random() < 0.2` branch and uniform integer candidate selection. With seed 42, 10,000 actual selections produced 2,024 exploration draws and candidate-index counts `[691, 683, 650]`. These observations support the implementation together with the inspected branch; they are not a proof of a random generator's distribution.
- Left/right orientation is randomized but biased: `select.py:298` samples 0, 1, or 2 because `endpoint=True`; only 1 keeps canonical order. The same 10,000 selections produced 3,393 canonical and 6,607 reversed presentations. Specs and design do not explicitly require equal orientation probabilities, so this is a non-blocking observation rather than an invented 50/50 acceptance criterion.
- No planning contradiction prevents determining required behavior. The design's draft resume examples differ from the implemented syntax, but the specification leaves argument syntax open. V-002 addresses the concrete public-help failure; any deliberate change to the planned command syntax should also be reconciled by the planning workflow.

## Checks

| Command or inspection | Result | Notes |
| --------------------- | ------ | ----- |
| `openspec status --change train-hotel-preference --json` | PASS | Resolved repo-local change root and schema; all planning artifacts present. |
| `openspec instructions apply --change train-hotel-preference --json` | PASS | Read every returned context file; 17 tasks checked, treated as claims. No previous verification report existed. |
| `openspec validate train-hotel-preference --type change --strict --json --no-interactive` | PASS | One valid change, zero issues. |
| `conda run -n hotel-model python -m pytest tests/test_data.py tests/test_features.py tests/test_fitting.py tests/test_fit_failures.py tests/test_select.py -q` | PASS | Focused data, mathematical, optimizer-failure and selection checks. |
| `conda run -n hotel-model python -m pytest` | PASS | `228 passed in 0.40s`; the dedicated conda environment was used throughout. |
| `conda run -n hotel-model hotel-model train --help` | FAIL | Advertises `--resume session.json`, which the parser rejects (V-002). |
| README commands executed in subprocesses | PASS | Read actual command lines, tokenize with `shlex`, substitute only a temporary state path, run the installed executable with `q`, then `a/q`, then `t/s/q`. All exit 0; final state has 3 events, 2 covered labels, 1 skip. Does not rely on the README regex test. |
| Answer saved, then injected `KeyboardInterrupt` in `train.fit` | FAIL | Exit is `interrupted`; JSON contains 1 label with coverage `0 + 0`. `load_state` rejects it (V-001). |
| Existing successful model, then failed second model save | FAIL | Answer is saved; JSON contains 2 labels with coverage `1 + 0`. `load_state` rejects it (V-001). |
| Real CLI SIGINT at input prompt | PASS | Signal delivered with `Popen.send_signal(SIGINT)` after observing the prompt. Exit 0, no answer, pending question unchanged; resume exits 0 with the same pending direction. |
| Real subprocess SIGKILL immediately upon entering fit | FAIL | Process exits -9 after saving the answer; JSON parses, but `load_state` rejects `0 + 0 != 1` (V-001). Only the verification-owned subprocess was terminated. |
| Injected `os.fsync` failure during second/model checkpoint | FAIL | Previous answer checkpoint remains syntactically intact but is rejected by application recovery (`1 + 0 != 2`, V-001). |
| Fingerprint and required-field mutation probes | PASS | Actual BOM CSV with reversed rows and an extra column has the same fingerprint; each of 11 individual required-field mutations differs. |
| Continuous vs. resumed execution | PASS | `a,b,t,s,a,q` versus `a,b,q` then `t,s,a,q`: identical event directions/labels, pending direction, RNG state, weights, loss, and model coverage. |
| 10,000 actual selections with recording RNG wrapper | PASS with observation | Default exploration 2,024/10,000; all legal candidate indices reached. Orientation bias is recorded above. |
| Exhaust a two-sample session by skipping its only pair | FAIL | No duplicate or cross-city question; termination explanation wrongly attributes absence to pool size or identical features (V-003). |
| CSV-valid positive subnormal area/price probes | FAIL | Each field accepts `5e-324`, then feature conversion raises `ValueError: math domain error` (V-004). |
| Manifest, environment, CLI, examples, README, implementation and tests | PASS with findings | Python src layout, NumPy/SciPy, console entry, training-only scope, templates, documentation and design formulas inspected. Findings detailed below. |

Runtime probes used temporary directories and the existing environment. No implementation, tests, planning artifacts, or task checkboxes were edited, and no installation or network access was required.

## Requirement Coverage

Paths in this table are repository-relative. Scenario names are translated for readability; each row maps to the current spec at the indicated line.

| Requirement / scenario | Evidence | Result |
| ---------------------- | -------- | ------ |
| Raw data contract (spec:7): required fields, identity, uniqueness, finite values, ranges, nine grades and units | `data.py:20-208`; `tests/test_data.py`; README sections 3 and 6; invalid data rejected before state creation in `test_training.py:299` | PASS for loading; downstream valid-extreme handling fails V-004 |
| Successful raw-data loading (spec:13) | `test_reads_valid_mixed_types`, `test_half_grade_mapping`, `test_all_nine_grade_names`; raw grade 4, age 2.5, rating 4.7 preserved | PASS |
| Reject invalid data (spec:17) | Parameterized identity/numeric/range tests, duplicate ID/column tests; `test_invalid_data_rejected_before_state_created` | PASS |
| Fixed/versioned preprocessing (spec:21): all eight formulas, ordering, constants, no city/pool normalization | `features.py:18-115`, immutable v1 configuration; formula/monotonicity/serialization tests; subnormal probe | FAIL: V-004 |
| City/pool-independent preprocessing (spec:36) | `test_preprocessing_independent_of_city_and_pool` evaluates equal raw inputs across cities and pools | PASS |
| Proportional price sensitivity (spec:40) | `test_price_ratio_sensitivity`: independent logarithm expectations and ratio approximately 3.44 | PASS |
| Same-city terminal comparison (spec:44): full raw display, normalized commands, invalid input, EOF, interruption | `train.py:62-79,262-311,339-381`; `diagnostics.py:103-122`; parameterized answer/display tests; real SIGINT probe | PASS at input; saved progress after fit interruption fails V-001 |
| Mixed-city data (spec:48) | Candidate tests, `test_two_city_session_asks_only_intra_city_pairs`, multi-city fit and counts tests | PASS |
| No effective candidates (spec:52) | `test_single_sample_city_reports_no_candidate`, `test_identical_feature_pairs_exhaust_candidates`; no cross-city fallback | PASS |
| Invalid input and exit (spec:56) | Invalid-input retention, q/EOF pending tests; installed executable q/resume demonstration | PASS |
| Labels and skips (spec:60): 1/0/0.5, skips excluded, unordered-pair removal | `fitting.py:84-125`; `train.py:350-381`; candidate direction-removal tests and integration event-prefix assertions | PASS |
| Tie versus unable-to-judge (spec:64) | `test_tie_skip_and_strict_are_recorded_distinctly`, `test_skips_do_not_enter_fitting`; actual README resume gives tie then skip | PASS |
| Display direction change (spec:68) | Flipped vectors/labels and independently calculated equal cross-entropies; displayed-direction integration test | PASS |
| Small-sample shared model (spec:72): shared 8D model, stable BCE, mean loss, positive L2, nonnegative bounds, all-label refits, failure retention | `fitting.py:144-292`; finite-difference gradient, fully separable/contradictory/tie/single-label tests; integration recomputed weights; intermediate checkpoint probe | PASS for fitting and reported optimizer failure; FAIL for interruption coverage V-001 |
| Single label and regularization (spec:78) | `test_single_strict_label_fits_finite_non_negative_weights`, separable/L2 tests and real single-answer CLI run | PASS |
| Joint multi-city fitting (spec:82) | `test_multi_city_shares_one_weight_vector`, two-city integration recomputation; exactly 8 saved weights | PASS |
| Optimizer failure (spec:86) | Optimizer exceptions/invalid outputs and preservation tests; integration failure handler persists a recoverable lagging model and resume refits | PASS for returned/handled optimizer failures; does not cover V-001 interruption |
| Informative active selection (spec:90): legal pairs, threshold 8, tradeoffs, uncertainty/magnitude/diversity, 20% exploration, seed 42, reproducibility, random orientation | `select.py:142-303`; phase-switch/formula/tradeoff/fallback tests; actual recording-RNG and uninterrupted/resumed comparisons | PASS for stated requirements; orientation bias noted separately |
| Cold start independent of untrained prediction (spec:94) | `test_cold_start_prefers_larger_difference_when_direction_repeated`, finite nonuniform cold scores, inspected branch before 8 labels | PASS |
| Prefer actual difference with equal prediction (spec:98) | `test_active_score_prefers_larger_difference_at_equal_uncertainty` evaluates norms 0.01 versus 1; selection uses those scores | PASS |
| Candidate exhaustion (spec:102) | Candidate exhaustion and no-repeat tests; two-sample exhaustion probe checks actual terminal text | FAIL: termination and removal work, explanation incorrect (V-003) |
| Recoverable local state (spec:106): snapshot/config/RNG/pending/events/model, no overwrite, atomic updates, answer-first persistence, protected restore | `state.py:163-488`; state round trips, invariants, no-overwrite, failure tests, independent signal/checkpoint/fsync probes | FAIL: intermediate checkpoint cannot be loaded (V-001) |
| Continue after exit (spec:110) | Pending-direction/config/RNG tests, actual README commands, independent continuous-versus-resumed comparison | PASS for orderly q/EOF/input SIGINT; broader interruption guarantee fails V-001 |
| Data-change protection (spec:114) | Fingerprint-change tests; integration changed-price rejection preserves original bytes; independent all-field fingerprint checks | PASS |
| Save failure (spec:118) | Existing replace-failure test preserves old bytes; independent failure of the second model save and its fsync leaves unrecoverable answer checkpoint | FAIL: V-001 |
| Diagnostics and sample preparation (spec:122): counts/cities/weights/coverage/loss, templates/examples/commands/units/formulas/conditions/core grades/limits | `diagnostics.py`; diagnostics tests; README read directly; examples parsed; commands executed; actual CLI help checked | PASS for diagnostics and README; FAIL for command help V-002 |
| Skip-only records (spec:128) | `test_diagnostics_with_only_skips`; independent skip-only exhaustion run has zero effective labels, untrained model, no fit-failure report or cross-entropy | PASS |
| Grade has no variation (spec:132) | `test_diagnostics_flags_feature_without_comparison_information`; identifies grade and rejects the interpretation that the user does not care | PASS |
| Proposal scope/non-goals | `pyproject.toml`, `cli.py`, all source modules and README: one training console entry, no scoring/recommendation/web/service/city parameters | PASS |
| Additional design constraints and tasks 1.1-6.3 | Fixed SciPy L-BFGS-B/analytic gradient; exact selection scoring and tolerances; preprocessing/config versions; immutable resumed config; snapshot fingerprint; answer-before-fit order; single-process restriction; documentation/example execution. Task 4.2 interruption claim fails; draft resume syntax differs; task completion is not accepted as proof. | FAIL: V-001, V-002; remaining checked work has inspected or executable evidence |

## Findings

### V-001: Answer-first checkpoint cannot be resumed before the model checkpoint

- Category: implementation
- Severity: high
- Requirement: Spec lines 46, 108-120 (saved progress, recoverable state, save failure); design decisions 4 and 6 (resume must refit uncovered labels); tasks 4.2 and 4.3.
- Evidence: `src/hotel_model/train.py:376` appends events and persists without changing model coverage. `src/hotel_model/state.py:419` rejects any state where `labels_covered + lagging_labels != label_count`. Starting from an untrained session, interrupting `train.fit` after input `a` leaves one saved label and model coverage `0 + 0`; `load_state` raises `StateError`. A real verification subprocess killed with SIGKILL upon entry to fit leaves the same state. Starting from one successfully fitted label, failure of the second model checkpoint (including injected `os.fsync` failure) leaves two saved labels with model coverage `1 + 0`, also rejected.
- Expected: Every committed answer checkpoint must preserve the previous successful model and be independently recoverable. An interrupted fit or failed model save must leave coverage behind by the exact number of uncovered labels, and resume must refit them before asking another question.
- Actual: The raw JSON is syntactically readable, but the application refuses to load it. The user cannot resume saved progress through the supported CLI. Normal optimizer-failure tests pass only because `_apply_fit` subsequently writes corrected lagging metadata.
- Repair guidance: Make the answer checkpoint internally consistent before its first atomic save, retaining the last successful weights/loss/covered count and representing newly uncovered labels accurately. Ensure every intermediate checkpoint loads and resumes safely when the second save never happens. Add meaningful checks at both save stages and during interrupted fitting; do not rely solely on a handled optimizer failure.
- Acceptance checks: Independently load the file immediately after saving first and later valid answers, before fitting; assert previous model preservation and exact lagging count. Interrupt fitting with SIGINT/KeyboardInterrupt and kill a verification-owned process after the answer save; resume must refit saved labels and never repeat answered pairs. Fail the second save at fsync and replace boundaries; recovery must succeed from the answer checkpoint. Fail the answer save; the preceding state must remain unchanged and loadable. Run `conda run -n hotel-model python -m pytest` after focused recovery checks.
- Repair status: pending
- Repair notes: None; first independent verification.
- Reverification notes: Confirmed through four independent executions: interrupted fit, real SIGKILL, failed model persistence, and model-save fsync failure.

Minimal reproduction, executable in the dedicated environment with `conda run -n hotel-model python -c '<code>'`:

```python
import tempfile
from pathlib import Path
from unittest.mock import patch
from hotel_model import train
from hotel_model.state import load_state
from hotel_model.train import ScriptedConsole, TrainOptions, run_training

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "session.json"
    with patch.object(train, "fit", side_effect=KeyboardInterrupt):
        run_training(
            TrainOptions(path, Path("examples/hotels_example.csv")),
            console=ScriptedConsole(["a"]),
        )
    load_state(path)  # Currently raises: model coverage 0 + 0 != 1.
```

### V-002: CLI help advertises an invalid resume command

- Category: implementation
- Severity: low
- Requirement: Spec line 126 (training command instructions); design decision 1 (`--help` and draft resume commands); task 5.1.
- Evidence: `src/hotel_model/cli.py:35` prints `hotel-model train --resume session.json [--data hotels.csv]`, while lines 39-40 define `--state` separately and `--resume` as a boolean. Running the advertised command with a valid temporary state exits 2 with `unrecognized arguments: <state path>`. `cli.py:51` additionally says `--resume` should specify the state file when both state and data are omitted. README's explicit `--resume --state session.json` commands all actually ran successfully.
- Expected: Public help and missing-argument errors must describe a command accepted by the parser.
- Actual: The resume example in `train --help` fails before loading the state. README's repaired commands do not repair this other documentation surface.
- Repair guidance: Make public help and argument diagnostics consistent with the supported resume syntax and verify help examples by execution. The spec does not require positional resume syntax; the design labels its entry examples as draft, so the difference alone is not a specification violation. Reconcile those planning examples through the planning workflow if retaining the current syntax, or implement the planned syntax coherently if that remains the intended interface.
- Acceptance checks: `conda run -n hotel-model hotel-model train --help`; run its actual resume example against a state created from the example CSV, both with and without optional `--data`. Omitted state must name the argument that actually supplies it. Recheck the README commands without relying on regex matches alone.
- Repair status: pending
- Repair notes: None; first independent verification.
- Reverification notes: Help output inspected and advertised syntax independently executed; actual README syntax passes.

### V-003: Exhaustion message attributes completion to the wrong cause

- Category: implementation
- Severity: low
- Requirement: Spec lines 102-104, candidate-exhaustion scenario; tasks 3.1 and 5.1.
- Evidence: `src/hotel_model/train.py:332` emits the same explanation for every empty candidate set: cities have fewer than two samples or all remaining pairs have identical features. In an independently executed session containing two distinct same-city samples, skipping the sole pair produces that message even though the candidate set was exhausted by processing it. `tests/test_training.py:239` only asserts the generic no-candidate phrase and `stop_reason == "exhausted"`.
- Expected: When all eligible pairs were answered or skipped, explain that the questions are exhausted and end normally without repetition or cross-city substitution.
- Actual: Normal termination, no repetition and no cross-city substitution work, but the terminal explanation attributes completion to input-pool problems rather than processed pairs.
- Repair guidance: Give an accurate exhausted/no-remaining-questions explanation for a consumed candidate pool while preserving the initial no-effective-candidate behavior.
- Acceptance checks: A two-sample same-city session with distinct features, followed by `a`, `t`, or `s`, must identify question exhaustion. Separate single-sample-city and identical-feature initial pools must still end normally with an accurate explanation. Assert the reason in rendered output, not only the outcome enum.
- Repair status: pending
- Repair notes: None; first independent verification.
- Reverification notes: Reproduced by skipping the only pair of a two-sample session restored from a valid checkpoint.

### V-004: Valid positive subnormal values fail the logarithmic transform

- Category: implementation
- Severity: low
- Requirement: Spec lines 9 and 23-34 (finite positive area/price and v1 natural-log transforms); task 1.3.
- Evidence: `src/hotel_model/features.py:105` and line 106 divide positive area/price by 25/300 before taking the logarithm. Independently parsing a CSV with either `room_area_sqm=5e-324` or `nightly_price=5e-324` succeeds and preserves a finite positive float. The division underflows to zero and `to_features` raises `ValueError: math domain error`. `test_vector_is_finite_for_extreme_inputs` only reaches area `1e-6` and price `1e9`.
- Expected: All finite positive inputs admitted by the stated contract have a finite v1 area/price feature; the mathematical logarithm of this positive value relative to the fixed center is finite.
- Actual: A valid input passes the data contract but cannot be transformed, so training stops. This has limited practical impact for hotel data but is a real unbounded-positive-domain conformance defect.
- Repair guidance: Evaluate the same mathematical logarithmic formula in a numerically stable way for the admitted positive floating-point domain. Do not silently introduce a new minimum value without revising the input contract.
- Acceptance checks: Parse CSVs using the smallest representable positive float independently for area and price; feature conversion must be finite and agree with a stable logarithm reference. Recheck ordinary formula values, ratio sensitivity, monotonicity, and versioned configuration.
- Repair status: pending
- Repair notes: None; first independent verification.
- Reverification notes: Both fields were independently accepted by the CSV loader and failed in feature conversion.

## Next Action

Run `$openspec-e13-fix-change train-hotel-preference` (Codex), or `/openspec-e13-fix-change train-hotel-preference` (other agents), to repair the pending implementation findings. Then run independent verification again before archive. Any deliberate revision of the planned command interface belongs to `$openspec-update-change train-hotel-preference` with the planning model; verification did not revise planning artifacts.
