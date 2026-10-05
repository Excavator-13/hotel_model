# Implementation Verification

- Change: train-hotel-preference
- Schema: spec-driven
- Verdict: PASS
- Verified revision: `a1bff79d35b91f3f00c599ff300e520c10288719` (`1st fix train-hotel-preference`). Worktree was clean before verification; this report is the only verification edit.
- Verification date: 2026-10-05

## Summary

Independent verification passes. Strict OpenSpec validation succeeds, the full suite reports **244 passed**, and independent execution confirms all four prior findings are resolved. All eight normative requirements and all twenty named scenarios have sufficient inspected and executable evidence. No pending implementation, planning, or environment findings remain; the change is ready for archive.

The important recovery fix was independently exercised at both persistence stages, before first and later fitting, and after a real SIGKILL. Every committed answer checkpoint now loads with the exact uncovered-label count and retains the previous successful weights and loss. Resume refits before selecting another question. Help and README commands were executed against temporary state files; their behavior was not inferred from parser or regex checks.

Residual observations and verification limits:

- Left/right order is randomized but biased: `src/hotel_model/select.py:317` uses `integers(0, 2, endpoint=True)`, producing three integer outcomes, only one of which retains canonical order. The artifacts require randomization, but do not require equal probabilities. This remains a non-blocking observation, not an invented acceptance criterion.
- Real SIGINT at a subprocess input prompt and real SIGKILL after the answer save were exercised. Physical Ctrl-C through a terminal driver, power loss, and physical disk faults were not exercised. Fsync and replacement failures were injected at the actual filesystem-operation boundary; stronger disk-durability guarantees are not promised.
- Design decision 1's draft resume examples use positional syntax; the supported interface uses `--resume --state <file>`. Public help, README, and missing-argument diagnostics now consistently describe and execute the supported syntax. The specs leave argument spelling open, so this draft difference does not block verification.
- Shared-city weights, fixed curves, nine equally spaced grades, and correlated features remain deliberate model limitations documented in README. Training cross-entropy is explicitly a training diagnostic, not calibration or generalization evidence.

## Checks

| Command or inspection | Result | Notes |
| --------------------- | ------ | ----- |
| `openspec status --change train-hotel-preference --json` | PASS | Resolved repo-local change root, schema `spec-driven`, and action context. |
| `openspec instructions apply --change train-hotel-preference --json` | PASS | Read every context file and the existing report; all 17 checked tasks treated as claims, not proof. |
| `openspec validate train-hotel-preference --type change --strict --json --no-interactive` | PASS | One valid change; zero issues. |
| `conda run -n hotel-model python -m pytest tests/test_state.py tests/test_integration.py tests/test_features.py tests/test_training.py tests/test_select.py -q` | PASS | All 147 focused tests pass; inspected the new regression tests and their actual interruption/failure boundaries. |
| `conda run -n hotel-model python -m pytest` | PASS | `244 passed in 0.42s`. No official lint/type/build check is configured. |
| Independent first/later interrupted-fit probe | PASS | At entry to fitting, load the actual answer checkpoint; injected KeyboardInterrupt leaves coverage `0 + 1` or `1 + 1`. Later interruption preserves prior weights and loss exactly; resume reaches `1 + 0` or `2 + 0`. |
| Independent fsync/replace failures at answer and model saves | PASS | Begin with one fitted label and a saved pending question; fail each actual operation at the first or second resume save. Failed answer saves leave byte-identical prior files and no success message. Failed model saves leave two labels with coverage `1 + 1` and prior weights/loss. All four checkpoints load and resume; temporary files are cleaned. |
| Real verification-owned subprocess SIGKILL at entry to fit | PASS | Process exits -9 after the first answer save. Checkpoint loads with one uncovered label; resume refits and records a different pair, covering both labels. |
| Real installed CLI SIGINT at input | PASS | Wait for the rendered prompt, send SIGINT: exit 0, no event, checkpoint unchanged; resume preserves pending direction. |
| Actual README training commands | PASS | Tokenize all three quick-start commands with `shlex`; run installed executable in the dedicated environment with a temporary state path and `q`, then `a/q`, then `t/s/q`. All exit 0; final events are strict/tie/skip and two labels are covered. |
| Actual `hotel-model train --help` examples | PASS | Execute all three advertised examples after substituting temporary paths. All exit 0; `hotel-model train --resume` exits 2 and correctly names `--state`. |
| Independent exhausted-pool output | PASS | A two-sample same-city pool consumed with each of `a/t/s` reports question exhaustion, has one event, and does not claim the initial pool lacked candidates. Initial no-candidate tests pass separately. |
| Independent positive-subnormal CSV probes | PASS | Parse `5e-324` separately for area and price. Finite features are approximately -747.658947746 and 750.143854396; agree with a 60-digit Decimal computation of the specified ratio logarithm. |
| Independent uninterrupted/resumed comparison | PASS | `a,b,t,s,a,q` versus `a,b,q` followed by `t,s,a,q`: entire state dictionaries agree with a fixed clock, including pending direction, events, RNG, configs, weights, loss, and coverage. |
| Independent invalid-state CLI probes | PASS | Unsupported schema, unknown pending reference, and cross-city pending question each exit 2 without changing source bytes. Event/label/duplicate/weight/config invariants are also exercised by inspected state tests. |
| Independent data/fingerprint probes | PASS | Each of eight numeric fields rejects empty/NaN/inf with row and field. Each of eleven required-field mutations changes the fingerprint; row reversal does not. BOM and extra-column semantics are covered by tests. |
| Actual controlled cold/active selection | PASS | Two same-city tradeoff pairs with norms 0.01 and 1 and equal prediction/diversity: non-exploration selection chooses the larger pair at both 0 and 8 effective labels. |
| Source, manifest, docs, examples, and scope inspection | PASS | Python src layout, editable-environment configuration, NumPy/SciPy, single training entry, formulas, exact selection scoring, state save ordering, diagnostics, and documentation verified. |

Runtime probes were executed with `conda run -n hotel-model python -c '<probe>'`, using temporary directories and existing dependencies. No implementation, tests, planning artifacts, or task checkboxes were edited; no installation, network access, or external mutation was required. The first README probe had a verifier token-slicing error before launching the application; the corrected probe above ran successfully and supplied the recorded evidence.

## Requirement Coverage

Paths are repository-relative; spec line numbers refer to `specs/hotel-preference-training/spec.md` within this change. Each named scenario is listed separately.

| Requirement / scenario | Evidence | Result |
| ---------------------- | -------- | ------ |
| 原始酒店数据契约 (spec:7): identity, uniqueness, eight finite/ranged numbers, nine grades, units, direct renovation age | `src/hotel_model/data.py`; `tests/test_data.py`; full raw-field display and README data/comparison sections; independent all-numeric validation probe. | PASS |
| 成功读取原始数据 (spec:13) | `test_half_grade_mapping` checks grade 4 -> 半高档, age 2.5, rating 4.7 from raw CSV; example loading tests. | PASS |
| 拒绝不合法数据 (spec:17) | Required-column, duplicate-ID, empty identity/numeric, fractional grade, rating, negative time, nonpositive area/price, NaN/inf tests; `test_invalid_data_rejected_before_state_created`; loader validates the entire pool before new-session persistence. | PASS |
| 统一且版本化的特征预处理 (spec:21): fixed order, constants, eight v1 formulas, no city/pool normalization | `src/hotel_model/features.py:88`; inspected formula/monotonicity/config tests; subnormal CSV probe confirms stable equivalent logarithms. | PASS |
| 预处理不受城市和样本池影响 (spec:36) | `test_preprocessing_independent_of_city_and_pool` compares identical inputs across cities and sample pools. | PASS |
| 价格变换反映比例敏感 (spec:40) | `test_price_ratio_sensitivity` checks independent logarithm expectations and ratio approximately 3.44. | PASS |
| 同城终端比较 (spec:44): full original display, normalized a/b/t/s/q, invalid input, EOF, interruption | `src/hotel_model/train.py:270`, `diagnostics.py:103`; display/answer/normalization/EOF tests, real installed-CLI SIGINT and interrupted-fit probes. | PASS |
| 混合城市数据 (spec:48) | `test_two_city_session_asks_only_intra_city_pairs`, candidate sorting tests, multi-city shared fit and counts; no cross-city candidate construction. | PASS |
| 无有效候选 (spec:52) | `test_single_sample_city_reports_no_candidate`, `test_identical_feature_pairs_exhaust_candidates`; rendered no-candidate reason, normal termination, no cross-city fallback. | PASS |
| 无效输入及退出 (spec:56) | Invalid-input retention and q/EOF tests, real command runs; no current-question event and pending direction retained. | PASS |
| 偏好标签与跳过语义 (spec:60): 1/0/0.5, skip excluded, unordered pair removed | `fitting.py:84`, `train.py:366`, `select.py:146`; label conversion, no-repeat and both-direction exclusion tests. | PASS |
| 平局与无法判断 (spec:64) | `test_tie_skip_and_strict_are_recorded_distinctly`, `test_skips_do_not_enter_fitting`; actual README session persists 0.5 tie and null skip distinctly. | PASS |
| 展示方向变化 (spec:68) | `test_swapping_display_and_flipping_label_keeps_target` independently evaluates equal cross-entropies; `test_label_matches_displayed_direction`. | PASS |
| 小样本共享线性模型 (spec:72): shared eight weights, no extra parameters, mean BCE + positive L2, nonnegative bounds, all-label refits, untrained/failure semantics | `fitting.py:144`, `state.py:491`, `train.py:260`; finite-difference gradient, stable loss, regularization and constraint tests; integration recomputation and checkpoint probes; diagnostics disclaimers. | PASS |
| 单一标签及正则化 (spec:78) | `test_single_strict_label_fits_finite_non_negative_weights`, separable/contradictory/all-tie tests and actual single-answer CLI execution. | PASS |
| 多城市共同拟合 (spec:82) | `test_multi_city_shares_one_weight_vector`, two-city integrations and recomputed weights; persisted model has exactly eight shared weights. | PASS |
| 优化失败 (spec:86) | `tests/test_fit_failures.py` rejects optimizer failure, exception, NaN/inf/negative/wrong-shape output; integration stops after preserved answer and resumes with refit; earlier successful weights/loss and exact uncovered count retained. | PASS |
| 主动选择有信息的问题 (spec:90): legal pairs, cold threshold 8, tradeoff priority/fallback, uncertainty/magnitude/diversity, default exploration/seed, random orientation, reproducibility | `select.py:146,256,282`; inspected phase/formula/tradeoff/fallback/exploration/randomization tests and exact continuous/resumed state probe. Defaults 0.2 and 42 and uniform candidate index branch verified in source. | PASS; orientation bias recorded as non-blocking |
| 冷启动不依赖未训练预测 (spec:94) | Cold-score/finite-nonuniform-score tests and actual controlled norm-1 versus norm-0.01 tradeoff selection at zero labels. | PASS |
| 相同预测下优先有实际差异的候选 (spec:98) | Active-score test and actual controlled selection at eight labels with zero weights, equal direction diversity, and two tradeoff pairs of norms 0.01 and 1. | PASS |
| 候选耗尽 (spec:102) | `test_candidate_exhaustion_reports_processed_pairs` and independent a/t/s two-sample probe; accurate exhaustion text, one event, no remaining question or repeated/cross-city pair. | PASS; V-003 verified-resolved |
| 可恢复的本地训练状态 (spec:106): snapshot/fingerprint/configs/RNG/pending/events/latest model, no overwrite, atomic update, answer-first save, protected restore | `state.py:163,340,459,491`, `train.py:214,366`; round-trip/invariant/no-overwrite tests and independent checkpoint, failure, signal, fingerprint and resume probes. | PASS; V-001 verified-resolved |
| 退出后继续 (spec:110) | Pending-direction/config/RNG/no-repeat tests, actual README/help commands, real SIGINT recovery, and exact uninterrupted/resumed comparison. | PASS |
| 数据变更保护 (spec:114) | `test_data_change_is_rejected_on_resume` checks changed-price rejection and unchanged bytes; fingerprint tests and independent all-required-field mutation probe. | PASS |
| 保存失败 (spec:118) | Independent real fsync/replace failures at answer/model saves preserve loadable previous state; failed answer saves give no success message, failed model saves preserve answer and old weights/loss, and resume refits. | PASS |
| 训练诊断和样本准备说明 (spec:122): all counts/cities/weights/coverage/loss, template/example/commands/units/formulas/conditions/core-grade guidance/limits | `diagnostics.py`, inspected diagnostics tests, README, examples, `tests/test_readme.py`; actual public-help and README commands executed. | PASS; V-002 verified-resolved |
| 只有跳过记录 (spec:128) | `test_diagnostics_with_only_skips` checks zero labels, skip count, untrained state, no weights or loss; independent skip-exhaustion run confirms normal behavior. | PASS |
| 档次无变化 (spec:132) | `test_diagnostics_flags_feature_without_comparison_information` uses equal-grade pairs and checks grade warning plus rejection of the interpretation that the user does not care. | PASS |
| Proposal scope and non-goals | All source modules, console manifest and parser inspected; one training command, no new-hotel score/rank/recommendation/web/service/city-specific model. | PASS |
| Additional design constraints and tasks 1.1-6.3 | Python 3.11+ layout/dependencies and installed editable console; SciPy L-BFGS-B/analytic gradient/warm start; exact M/D/U scores/tolerances and stable sorting; full versioned config/fingerprint; single-process docs; both persistence stages and recovery; examples/docs and complete tests. Draft syntax difference discussed in Summary. | PASS |

## Findings

All findings below are retained historical defects from revision `e34dca5351c6c41b8e02ede8438ed360dbf83898`. Their prior repair notes are preserved verbatim. Current evidence resolves each one; there are no pending findings.

### V-001: Answer-first checkpoint cannot be resumed before the model checkpoint

- Category: implementation
- Severity: high
- Requirement: Spec lines 46, 108-120; design decisions 4 and 6; tasks 4.2 and 4.3.
- Evidence: Previously, the answer checkpoint retained stale uncovered-label metadata and was rejected by `load_state`. Current `train.py:400` updates the model through `state.py:491` in the same saved answer checkpoint.
- Expected: Every committed answer checkpoint is loadable, preserves the last successful model, records exact uncovered labels, and resumes by refitting before selecting another question.
- Actual: Independent first/later fit interruptions, real SIGKILL, and actual fsync/replace failures at both saves all leave loadable checkpoints; resume refits and processed pairs are excluded.
- Repair guidance: Completed; maintain this checkpoint invariant.
- Acceptance checks: Focused recovery tests, full pytest, and independent checkpoint/signal/fsync/replace probes recorded in Checks.
- Repair status: verified-resolved
- Repair notes: `state.py` adds `model_after_label` (keeps the last successful weights/loss/covered count and sets `lagging_labels` to the exact uncovered label count) and reuses it in the `model_after_fit` failure path. `train.py` `_record_answer` now writes that model snapshot in the same atomic save as the event, so every answer checkpoint is internally consistent before fitting. Tests added: `tests/test_state.py::test_answer_checkpoint_loads_before_first_fit`, `::test_answer_checkpoint_keeps_last_successful_model`, `::test_model_checkpoint_failure_keeps_answer_checkpoint` (parametrized over fsync and replace); `tests/test_integration.py::test_interrupted_first_fit_leaves_loadable_checkpoint_and_resume_refits`, `::test_answer_checkpoint_after_later_answer_keeps_previous_weights`, `::test_failed_answer_save_keeps_previous_checkpoint`, `::test_failed_model_save_keeps_answer_checkpoint_and_resume_recovers`. `conda run -n hotel-model python -m pytest` -> 244 passed. Independent probes: a real subprocess SIGKILL after the answer save leaves a loadable `0 + 1` checkpoint that resume refits and continues without repeating the answered pair; an injected `os.fsync` failure at the model checkpoint (4th save) leaves the answer checkpoint loadable and resume refits it; a failed answer save leaves the previous pending-question checkpoint byte-identical and loadable.
- Reverification notes: Rechecked on 2026-10-05 at a1bff79. First/later interrupted fits leave coverage 0+1 and 1+1; later fit keeps the independently captured prior weights and loss exactly. Each answer-save failure leaves byte-identical prior state; each model-save failure retains two labels with coverage 1+1. Real SIGKILL exits -9 and recovery covers saved labels without repeating the answered pair.

The prior minimal reproduction now succeeds:

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
    saved = load_state(path)
    assert saved.model.labels_covered == 0
    assert saved.model.lagging_labels == 1
    resumed = run_training(
        TrainOptions(path, resume=True), console=ScriptedConsole(["q"])
    )
    assert resumed.state.model.labels_covered == 1
    assert resumed.state.model.lagging_labels == 0
```

### V-002: CLI help advertises an invalid resume command

- Category: implementation
- Severity: low
- Requirement: Spec line 126; design decision 1; task 5.1.
- Evidence: Previously, help advertised a positional file after boolean `--resume`. Current `cli.py:34` advertises `--resume --state session.json`; `cli.py:53` correctly names the missing state argument.
- Expected: Public examples and errors describe syntax accepted by the parser and executable against a valid state.
- Actual: All three help examples execute successfully; both resume variants load saved state; missing-state error names `--state`; all three README examples also execute.
- Repair guidance: Completed; retain executable public examples.
- Acceptance checks: Execute examples from actual `hotel-model train --help` output and README, including optional CSV validation; missing-state error identifies `--state`.
- Repair status: verified-resolved
- Repair notes: `cli.py` `train --help` now uses `argparse.RawDescriptionHelpFormatter` and lists literal executable examples `hotel-model train --resume --state session.json` and `hotel-model train --resume --state session.json --data hotels.csv`, plus a line stating `--resume` is a switch and `--state` supplies the file. The duplicate missing-argument branch was collapsed to one message naming `--state`. Tests added: `tests/test_training.py::test_train_help_examples_are_parseable_and_use_state`, `::test_train_help_resume_examples_run_against_state_file`, `::test_resume_without_state_names_the_state_argument`. Executed every resume example from the help output against a state created from `examples/hotels_example.csv`, with and without `--data` (exit 0, no `unrecognized arguments`); `hotel-model train --resume` reports the `--state` argument. README resume commands still execute.
- Reverification notes: Independently tokenized actual help/README command lines and ran the installed executable with temporary path substitutions. Six examples exit 0; resume without state exits 2 and names --state. The design's draft spelling is recorded separately as non-blocking.

### V-003: Exhaustion message attributes completion to the wrong cause

- Category: implementation
- Severity: low
- Requirement: Spec lines 102-104; tasks 3.1 and 5.1.
- Evidence: Previously, a consumed pool displayed the initial no-candidate explanation. Current `select.py:176` checks the unprocessed pool and `train.py:342` renders the appropriate reason.
- Expected: A consumed candidate pool reports exhausted questions; initially ineffective pools report the appropriate lack of candidates; both terminate normally without repeated/cross-city questions.
- Actual: Independent two-sample sessions consumed with a/t/s each report exhaustion. Single-sample and identical-feature regression tests report no candidate separately.
- Repair guidance: Completed; retain separate reasons.
- Acceptance checks: Rendered-output tests and actual two-sample strict/tie/skip probes in Checks.
- Repair status: verified-resolved
- Repair notes: `select.py` adds `EXHAUSTED`/`NO_CANDIDATES` and `reason_for_empty_candidates`, which rebuilds candidates while ignoring processed records: an informative pair still exists means the pool was consumed. `train.py` `_select` prints "问题已耗尽" plus "不重复提问，也不构造跨城市问题" for that case and keeps the original no-effective-candidate text otherwise. Tests: `tests/test_select.py::test_empty_candidate_reason_distinguishes_causes`; `tests/test_training.py::test_candidate_exhaustion_reports_processed_pairs` (parametrized over `a`/`t`/`s`, asserting the rendered exhaustion text and the absence of the no-candidate phrase) with negative assertions added to the single-sample-city and identical-feature tests. Probes confirmed a two-sample session ended by `a`/`t`/`s` and a resumed two-sample session ended by `s` both report exhaustion, while single-sample and identical-feature pools still report no candidate.
- Reverification notes: Rechecked all three consumption answers against actual two-sample same-city CSV files; each has one event, no pending question, stop reason exhausted, correct exhaustion text, and no initial-pool explanation. Initial no-candidate test outputs were inspected and pass.

### V-004: Valid positive subnormal values fail the logarithmic transform

- Category: implementation
- Severity: low
- Requirement: Spec lines 9 and 23-34; task 1.3.
- Evidence: Previously, area/price division underflowed before log. Current `features.py:108` evaluates equivalent log differences; independent CSV probes admit and transform the smallest positive float.
- Expected: Finite positive area/price admitted by the contract produce finite v1 log features without changing the input domain.
- Actual: Area 5e-324 -> approximately -747.658947746; price 5e-324 -> approximately 750.143854396. Both agree with independent 60-digit Decimal ratio-log references. Ordinary formulas, monotonicity, proportional sensitivity, and versioned config checks pass.
- Repair guidance: Completed; preserve the specified mathematical transform and admitted domain.
- Acceptance checks: Smallest-positive-float CSV probes, ordinary feature/formula/monotonicity/ratio/config tests, full pytest.
- Repair status: verified-resolved
- Repair notes: `features.py` evaluates the same v1 logarithms as `log(x) - log(center)` (mathematically equal to `log(x / center)`, but the division no longer underflows a small positive input to zero before `math.log`). Test added: `tests/test_features.py::test_smallest_positive_subnormal_still_has_finite_log_feature` (parametrized for area and price), which parses `5e-324` through the CSV loader and compares the feature against the stable logarithm reference. Ordinary formula values, ratio sensitivity, monotonicity, city/pool independence and versioned-configuration tests still pass; `conda run -n hotel-model python -m pytest` -> 244 passed.
- Reverification notes: Rechecked both fields through CSV parsing before conversion, compared with Decimal.from_float(5e-324) / Decimal(center) at precision 60 followed by Decimal.ln, and asserted all eight transformed values are finite.

## Next Action

Independent verification passed. The change is ready for `$openspec-archive-change train-hotel-preference` (Codex), or `/openspec-archive-change train-hotel-preference` (other agents). This verification run does not archive the change.
