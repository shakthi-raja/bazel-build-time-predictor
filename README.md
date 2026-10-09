# Early-Phase Bazel Build Time Prediction and CI Risk Triage

Machine learning mini-project for **UE24CS352A Machine Learning**, Problem Statement 60.

| | |
|---|---|
| **Team** | Shakthi Raja G P (PES1UG25CS843) |
| | Anuvidya (PES1UG25CS804) |
| **Section** | L |
| **Problem** | 60, Predicting Bazel Build Times Using Machine Learning |
| **Benchmark** | Nadeem and Raza, Stanford CS229, 2021 |

---

## What this project does

Large Bazel builds are expensive and unpredictable. A build that looks small can invalidate a large part of the action graph and run for tens of seconds, which causes CI timeouts and wasted compute. The original Stanford project predicted CPU time from two commit-level fields: the dominant changed directory and the dominant file type.

We reproduce that baseline, then ask a more useful question:

> At the end of Bazel's analysis phase, before any compilation action starts, can we predict the CPU time and the timeout risk of this build?

The answer is yes, to a useful degree. Our selected model reaches an **R2 of 0.702** and a **test RMSE of 3,159 ms**, compared with an R2 of -0.044 and a test RMSE of 5,911 ms for the baseline. That is a **46.6% reduction in RMSE**.

We also add a three-tier **CI risk guard** that turns the prediction into an action: run locally, use a standard CI worker, or use a high-memory worker with a longer timeout. The risk classifier reaches **78.0% accuracy** and **0.719 macro F1**, with **zero High-risk builds classified as Low**. High-risk recall is **68.4%**.

---

## What is available, and when

This is the most important design decision in the project.

| Stage | Signals | Used by the model? |
|---|---|---|
| Before the build starts | dominant changed directory, dominant file type | Yes |
| End of analysis, before compilation | packages loaded, targets configured, action graph size | Yes |
| After the build finishes | actions executed, cache hit ratio, elapsed time | **No** |

The post-execution fields are very predictive. Adding `actions_executed` raises the test R2 from 0.702 to 0.927. But a CI scheduler does not know that value before the build runs, so using it would be target leakage. We keep that experiment in the code as a reference upper bound and never use it in the deployed model.

The pure commit-metadata model is also included as the Stanford baseline. It confirms the paper's finding that two categorical fields are not enough.

---

## Dataset

The released Stanford dataset contains 500 consecutive builds of `bazelbuild/bazel` on macOS.

- `data/raw/InputData.csv`: commit id, dominant changed directory, dominant file type
- `data/raw/CPUTimes.csv`: CPU build time in milliseconds
- `data/raw/Build_Info/`: 500 Build Event Protocol JSON streams
- `data/processed/bazel_build_dataset.csv`: rebuilt table with analysis-phase counts

The split follows the paper: first 300 builds for training, next 100 for validation, final 100 for testing. The test set is never used for model selection.

---

## Method

1. Parse each BEP stream for the final `packages loaded` and `targets configured` counts at the end of the analysis phase.
2. Read the total number of actions in the build graph from Bazel's progress output.
3. Add the commit's dominant directory and file type.
4. Derive log counts and ratios: actions per target, targets per package, cached-build flag, and change-type flags.
5. Compare Ridge, Random Forest and HistGradientBoosting on the validation split.
6. Select **HistGradientBoosting** and evaluate it once on the test split.
7. Train a balanced Random Forest classifier for the three risk tiers.

No post-execution metric is passed to either model.

---

## Results

### Regression

| Model | Train RMSE | Validation RMSE | Test RMSE | Test MAE | Test R2 |
|---|---:|---:|---:|---:|---:|
| Stanford baseline (prefix and type) | 5,375 | 5,960 | 5,911 | 4,351 | -0.044 |
| Ridge | 3,647 | 4,172 | 3,905 | 2,525 | 0.544 |
| Random Forest | 1,499 | 2,993 | 3,475 | 1,866 | 0.639 |
| **HistGradientBoosting (selected)** | **904** | **2,984** | **3,159** | **1,753** | **0.702** |
| Post-execution reference, not deployable | 265 | 1,315 | 1,566 | 734 | 0.927 |

The selected model is chosen on validation RMSE. The test column is reported once for that model. The post-execution reference is included only to show how much signal is lost when we refuse to use future information.

### CI risk guard

| Tier | Rule | Suggested action |
|---|---|---|
| Low | CPU time below 2.5 s | Run locally or on a fast feedback queue |
| Medium | 2.5 s to 10 s | Standard CI worker |
| High | Above 10 s | High-memory worker, extended timeout |

Test confusion matrix, rows actual and columns predicted:

| | Low | Medium | High |
|---|---:|---:|---:|
| **Low** | 52 | 11 | 2 |
| **Medium** | 0 | 13 | 3 |
| **High** | 0 | 6 | 13 |

- Accuracy: 78.0%
- Macro F1: 0.719
- Low precision: 100%, Low recall: 80.0%
- High precision: 72.2%, High recall: 68.4%
- High predicted as Low: 0
- High predicted as Medium: 6

The zero in the bottom-left cell is the safety property we care about: no build that actually took more than 10 s was routed to the local/fast lane. The six High builds predicted Medium are the main cost of the balanced classifier, and we state that clearly instead of claiming perfect recall.

---

## Project layout

```text
bazel-build-time-predictor/
├── data/
│   ├── raw/                         # original 500 BEP logs and CSVs
│   └── processed/
│       └── bazel_build_dataset.csv  # rebuilt table with analysis-phase features
├── docs/
│   ├── writeup.docx                 # editable two-page report
│   ├── writeup.pdf                  # two-page report
│   ├── presentation.pptx            # editable slide deck
│   └── presentation.pdf             # eleven-slide deck
├── models/
│   ├── preprocessor.joblib
│   ├── regressor.joblib
│   └── risk_classifier.joblib
├── notebooks/
│   └── bazel_build_time_prediction.ipynb
├── reports/
│   ├── figures/
│   ├── metrics.json
│   ├── regression_metrics.csv
│   └── test_predictions.csv
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
│   └── test_project.py              # data, leakage and CLI smoke tests
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Run the project

### 1. Rebuild the dataset

```bash
python src/data_loader.py
```

This parses all 500 BEP files and writes `data/processed/bazel_build_dataset.csv`.

### 2. Train and evaluate

```bash
python src/train.py
python src/evaluate.py
```

`train.py` writes the models, `reports/metrics.json`, `reports/regression_metrics.csv` and `reports/test_predictions.csv`. `evaluate.py` writes the figures used in the report and slides.

### 3. Open the notebook

```bash
jupyter notebook notebooks/bazel_build_time_prediction.ipynb
```

The notebook is already executed and contains the same numbers as the report.

### 4. Demo the CLI

The command below estimates the build for a real held-out commit. Commit 1 is a large site build and commit 2 is a small cached build.

```bash
python src/predict.py --sample 1
python src/predict.py --sample 2
```

You can also supply the analysis-phase counts directly:

```bash
python src/predict.py --prefix src/main --type JAVA \
  --packages-loaded 4 --targets-configured 1462 --total-actions 2883
```

Example output:

```text
================================================================
 BAZEL BUILD TIME AND RISK ESTIMATE
================================================================
 Commit profile       : src/main / JAVA
 Analysis phase       : 4 packages, 1462 targets, 2883 actions
 Predicted CPU time   : 11,225 ms (11.22 s)
 Risk tier            : High
 Class probabilities  : Low 0.5% | Medium 7.2% | High 92.3%
 Recommended action   : Large build. Use a high-memory worker and extend the timeout.
================================================================
```

### 5. Run the smoke tests

```bash
python -m unittest discover -s tests -v
```

The tests check the 500-row dataset, the 300 / 100 / 100 split, the absence of post-execution features, the saved models and the CLI prediction path.

---

## Limitations

- The dataset has no machine load, cache state, or exact changed-file list, so some builds remain hard to predict. The test RMSE of 3,159 ms is useful for triage but not precise enough for billing.
- The analysis-phase counts come from the BEP stream of the same build. They are available at the end of analysis, but the public dataset does not expose a clean pre-execution API. In a production integration, Bazel would need to emit these values at the analysis barrier.
- The `total_actions` value is read from Bazel's progress output. It is the action graph size, not the number of actions that ran. The number that ran is deliberately excluded.
- The risk thresholds (2.5 s and 10 s) are operational choices. A different team could choose different cut-offs and would get a different class balance.
- The classifier is trained on 500 builds from one codebase. A new codebase would need a short calibration period.

---

## References

1. M. U. Nadeem and S. Raza. *Predicting Bazel Build Times Using Machine Learning.* Stanford CS229 final project, 2021.
2. Google. *Build Event Protocol.* https://bazel.build/remote/bep
3. K. Wang et al. *Smart Build Targets Batching Service at Google.* ICSE-SEIP, 2021.
