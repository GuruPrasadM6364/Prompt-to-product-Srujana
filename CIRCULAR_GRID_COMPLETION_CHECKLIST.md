# EcoFlow Campus: Requirements and Dataset Checklist

## Scope and how to read this checklist

EcoFlow Campus is the current product name; the original prompt and supplied
problem statement refer to the project as Circular Grid. This checklist
compares the supplied problem statement, the seven workbooks in
`P2P_DATASET/`, the accompanying data guide, and the current repository.
Statuses describe evidence found in the current workspace, not the team's
intentions:

- **DONE**: directly verified in the files or by a completed check.
- **PARTIAL**: some code or evidence exists, but an important requirement is
  incomplete, disconnected, or not demonstrated end to end.
- **NOT DONE**: no working implementation or deliverable was found.
- **BLOCKED**: the current setup prevents reproducible validation.

The supplied data guide explicitly describes the data as a **synthetic /
hypothetical benchmark**. Any impact, forecasts, or operational conclusions
must be presented as benchmark results or simulations, not real campus
measurements or proven real-world savings.

## Dataset inventory

All seven Excel workbooks are present in `P2P_DATASET/`. Their sheet schemas
and row counts were inspected and match the corresponding CSV row counts in
`circular_grid/data/`.

| Dataset | Workbook rows | CSV rows | Intended use | Status |
|---|---:|---:|---|---|
| `buildings.xlsx` | 8 | 8 | Building IDs, type, capacity, solar, composters | **DONE**: present and count matches |
| `calendar.xlsx` | 120 | 120 | Day type, events, and weather context | **DONE**: present and count matches |
| `daily_summary.xlsx` | 960 | 960 | Daily cross-checks and waste/resource summaries | **DONE**: present and count matches |
| `hourly_streams.xlsx` | 23,192 | 23,192 | Meter, solar, and footfall time series | **DONE**: present and count matches |
| `manual_logs.xlsx` | 415 | 415 | Alias/unit normalization and operational notes | **DONE**: present and count matches |
| `waste_events_train.xlsx` | 6,314 | 6,314 | Labeled classifier training and evaluation | **DONE**: present and count matches |
| `waste_events_test.xlsx` | 1,601 | 1,601 | Unlabeled predictions only; no accuracy claims | **DONE**: present and count matches |

The train workbook has the `category` label column; the test workbook does not.
Counts and schemas match, but this check does not prove a clean-clone
re-import of the workbooks or exact cell-by-cell equality between every
workbook and CSV.

## Problem-statement checklist

### 1. Multi-source ingestion, fusion, and data quality

- [x] **DONE** — All seven input workbooks and the corresponding CSV row counts
  are present and match.
- [x] **DONE** — An audit report exists at
  `circular_grid/data/quality_report.json`, reporting such items as 184
  duplicate readings, missing values, alias counts, and imputed counts.
- [ ] **PARTIAL** — Hourly ingestion parses timestamps, normalizes building
  names, deduplicates, flags gaps, and interpolates some meter gaps in
  `circular_grid/src/ingestion/audit.py`.
- [ ] **PARTIAL** — Deduplication chooses rows by missing-value count and
  receive time; the requested neighbor-consistency rule is not demonstrated.
  The overall row imputation flag is set to `False`, even when individual
  values were interpolated, so estimated values can be mislabeled.
- [ ] **PARTIAL** — Daily summaries are compared to hourly totals and
  conflicts are written to a CSV, but the documented “use daily as ground
  truth” behavior does not update the fused hourly output. Manual logs are
  alias-checked, but their values and notes are not fused into the readings
  with per-value source precedence and confidence.
- [ ] **PARTIAL** — Some audit checks are permissive: for example, the solar
  check can pass without verifying the exact expected set. Make every
  assertion test the specified fact, and fail the audit when a required
  condition is not met.
- [ ] **BLOCKED** — A clean `make data` is not yet reproducible from the
  supplied files: `P2P_DATASET/` contains `buildings.xlsx`, etc., while
  `circular_grid/config.py` defaults to `circular_grid/data/raw` and expects
  different `media_*.xlsx` filenames. Point the configuration at the actual
  seven filenames or provide a documented environment configuration that
  works without renaming the source files.

### 2. Waste classification at the edge

- [ ] **PARTIAL** — `circular_grid/src/classifier/train.py` defines three model
  candidates, missing-value imputation, class balancing, a time holdout, model
  export, and predictions. No trained model, classifier results report, or
  predictions CSV was found in the current workspace.
- [ ] **PARTIAL** — The existing classifier tests skip without a trained model;
  two tests were skipped in the latest run. Run training from the supplied
  workbook and record holdout accuracy, macro-F1, per-class recall, confusion
  matrix, model size, and p50/p95 latency.
- [ ] **NOT DONE** — Cross-validation uses `StratifiedKFold` over a
  time-ordered dataset. This is not a time-based CV protocol and may train on
  later records while validating earlier ones. Replace it with an explicitly
  chronological validation design and document the chosen folds.
- [ ] **PARTIAL** — The API's `/api/classifier/classify` and classifier feed use
  hand-written sensor rules instead of loading the selected trained model.
  Confidence values are hard-coded by rule, not calibrated model
  probabilities. Do not label these as measured classifier accuracy.
- [ ] **NOT DONE** — No `waste_events_test_predictions.csv` was found. Generate
  it from the chosen trained model with `event_id` and `predicted_class`
  (confidence may be an additional column); make no accuracy claim on this
  unlabeled dataset.
- [ ] **PARTIAL** — A low-confidence path is tested for the trained inference
  service, but it is not wired into the API's rule-based classifier response.

### 3. Six- to twelve-hour forecasting

- [ ] **PARTIAL** — Forecasting code and baseline/evaluation routines exist.
  Recent changes keep features and baseline rows grouped by building and add
  assertions for chronological horizon behavior.
- [ ] **PARTIAL** — Forecast validation now has targeted tests, but no
  forecast-results report or trained model files were found. Run the full
  evaluation on the supplied hourly data and retain the measured MAE, sMAPE,
  spike precision/recall, training time, inference latency, and model size.
- [ ] **PARTIAL** — The API forecast endpoint currently constructs repeated
  values from recent history; it does not load the trained forecast model or
  return real calendar event overlays or model-derived uncertainty intervals.
  The UI therefore must not be described as displaying a validated model
  forecast until this is connected.
- [ ] **NOT DONE** — Waste generation is not included as a forecast target.
- [ ] **PARTIAL** — Check exact temporal alignment, timezone handling, missing
  hours, and train/validation feature construction against real workbook
  records. Missing readings should be handled explicitly, not silently
  converted to valid zero demand.

### 4. Offline operation and data integrity

- [ ] **PARTIAL** — SQLite WAL edge-buffer helpers and reconciliation/outage
  simulation exist, and related tests pass. The edge buffer is not integrated
  with an API upload/acknowledgement path.
- [ ] **PARTIAL** — The outage replay test verifies aggregate totals for a
  simulation, not a complete per-building live outage/reconnect workflow.
  Add retries, batch acknowledgements, duplicate uploads, out-of-order
  readings, and conflict tests.
- [ ] **PARTIAL** — The dashboard API derives building status from the
  historical CSV, can omit a building with no rows, and does not use the edge
  reconciliation status store. It does not yet guarantee that all eight
  buildings remain visible as online, stale, or offline.

### 5. Root-cause explanations and recommendations

- [ ] **PARTIAL** — An anomaly detector and attribution/recommendation logic
  exist in `circular_grid/src/anomaly/detector.py`.
- [ ] **NOT DONE** — The `/api/anomalies` endpoint returns an empty list and is
  not connected to the detector. No operational anomaly/recommendation screen
  is present in the frontend.
- [ ] **NOT DONE** — Demonstrate top causes with the exact supporting
  calendar, weather, meter, and manual-log evidence; show confidence and
  distinguish estimated savings from realized savings.

### 6. Occupant incentives and impact evaluation

- [ ] **PARTIAL** — A minimal credit ledger, occupancy normalization, and
  load-shift helper now exist in `circular_grid/src/incentive/credits.py`.
- [ ] **NOT DONE** — Credits are in-memory only; there is no persistent
  auditable ledger connected to meter/bin data or a leaderboard UI.
- [ ] **NOT DONE** — Define and run an impact evaluation with a stated
  baseline, comparison method, controls, sample-size limitations, and
  simulation labels. Do not claim a causal behavior change from synthetic
  benchmark data.
- [ ] **PARTIAL** — Anti-gaming unit tests exist, but test that credits cannot
  be earned without evidence and that occupancy/event normalization is
  grounded in the provided data rather than fixed example values.

### 7. Cost, scale, and failure modes

- [ ] **PARTIAL** — A scale-test script exists, but it generates synthetic
  campus data and no benchmark result file is present. Clearly label such
  measurements synthetic and do not imply they are real device measurements.
- [ ] **NOT DONE** — No line-item per-building sensor/compute/power budget or
  total hardware estimate was found.
- [ ] **NOT DONE** — The `/api/cost-scale` endpoint returns empty placeholder
  values.
- [ ] **PARTIAL** — Some mitigations have code/tests (outage replay,
  low-confidence fallback, anti-gaming helpers). No implemented/tested
  sensor-drift detector or model-degradation/retraining trigger was found.
- [ ] **PARTIAL** — Failure-mode documentation should name the trigger,
  detection signal, mitigation, test evidence, and remaining limitations for
  each failure mode.

### 8. Dashboard and submission deliverables

- [ ] **PARTIAL** — The frontend has a campus overview, building detail, and
  waste classifier view. It does not yet include the requested anomaly,
  circular allocation, data-quality/connectivity, incentives, and cost/scale
  screens.
- [ ] **PARTIAL** — Some APIs exist for the missing screens, but several return
  hard-coded zero/empty values. Do not display these as measured metrics.
- [ ] **NOT DONE** — No 1440px screenshot package, standardized three-page
  report PDF, or 2.5-minute video/demo script was found.
- [x] **DONE** — A prompt workbook exists at `AI_PROMPT_WORKBOOK.txt`; confirm
  it meets the submission portal's requirements for prompt evolution and
  verified public chat share links.
- [ ] **PARTIAL** — README and Make targets exist. Dependency requirements
  currently use version ranges rather than a fully locked reproducible
  environment. The `make data` path mismatch must be fixed before a fresh
  setup can be called reproducible.

## Current validation snapshot

- **DONE** — The current Python suite: 60 passed, 2 skipped. Both skipped
  classifier tests require trained model artifacts that are not present.
- **DONE** — Frontend TypeScript typecheck and production build passed.
- **PARTIAL** — The frontend build still emits Vite/plugin deprecation notices
  and warns that the JavaScript bundle exceeds 500 kB.
- **DONE** — The local `make run` target binds to `127.0.0.1` by default.
- **NOT DONE** — No end-to-end run from the seven original workbooks through
  training, prediction, API, and dashboard was verified.

## Prioritized completion checklist

1. [ ] Fix `RAW_FILES`/`UPLOAD_DIR` so `make data` reads the seven supplied
   workbooks in `P2P_DATASET/` without renaming or replacing them.
2. [ ] Run the audit from a clean state; assert every specified dataset fact
   correctly and verify every transformation/count in the quality report.
3. [ ] Fix chronological classifier cross-validation; train from the supplied
   train workbook; create real measured result artifacts and test predictions.
4. [ ] Connect classifier and forecasting models to the API; replace
   rule-generated confidence and repeated-value forecasts or label them
   unmistakably as demos.
5. [ ] Connect source-level provenance, correct per-cell imputation flags,
   manual-log evidence, and actual daily cross-check resolution.
6. [ ] Connect the edge buffer/reconciliation path and guarantee all eight
   buildings appear with truthful status/estimate labels.
7. [ ] Connect anomaly findings and evidence-backed recommendations to the
   dashboard; remove or clearly mark all zero/empty placeholders.
8. [ ] Implement the persistent incentive ledger, anti-gaming audit trail,
   and a clearly labeled impact simulation.
9. [ ] Complete and record hardware budget, tested failure modes, and
   explicitly labeled 8-to-32-building synthetic scale measurements.
10. [ ] Capture the required screenshots, produce the three-page report and
    demo script/video, then verify every claim against saved artifacts.

## Additional features that would make the project distinctive

Prioritize features that increase trust and operational value over adding more
screens:

1. **“Why should I trust this value?” lineage inspector** — show the dataset,
   original row/time, cleaning/unit conversion, reconciliation rule, and
   measured/imputed/estimated label behind every dashboard metric and
   recommendation.
2. **Discrepancy review workflow** — compare hourly totals, daily summaries,
   and relevant manual logs side by side; let an operator record the
   resolution while preserving original readings and an audit trail.
3. **Water-energy coupling** — estimate pumping/treatment energy from water
   volume using the guide's 0.85 kWh per 1,000 L assumption, label it as an
   estimate, and surface water leaks that also increase energy use.
4. **Resource digital-twin scenario planner** — let facilities staff test
   scenarios such as a scheduled event, a water restriction, or a solar
   outage, and show forecast ranges and assumptions rather than a single
   guaranteed outcome.
5. **Model abstention and human feedback** — send low-confidence waste events
   to manual sorting, collect corrected labels, and report how feedback
   changes future model performance without leaking test labels.
6. **Sensor health and drift monitoring** — detect stuck, drifting, or
   physically implausible sensors; distinguish sensor failure from actual
   resource anomalies.
7. **Intervention evidence tracking** — record an action, owner, date, and
   expected savings; later compare against a weather/calendar-adjusted
   baseline and report uncertainty without claiming causation from a
   simulation.

The strongest differentiator is a single evidence trail connecting all seven
datasets to each operational recommendation, with uncertainty and unresolved
conflicts visible to the facilities team.
