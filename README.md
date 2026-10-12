# SAFE-Gait

**One-shot supervised residual personalization for cross-condition gait prediction**

SAFE-Gait investigates a simple but practically important question:

> **Can one labeled calibration gait pass reveal subject-specific residual information that transfers to a different, independently acquired gait pass?**

Rather than asking whether a more complex personalization model is universally better, SAFE-Gait focuses on **when one-shot personalization transfers, when the benefit weakens, and when personalization causes negative transfer**.

---

## Overview

Population gait-prediction models can achieve good average performance while still producing systematic errors for individual unseen subjects. SAFE-Gait treats part of this error as potentially **subject-specific and transferable**, rather than as pure noise.

For an unseen subject, the framework uses:

1. a **frozen population model**,
2. one **supervised calibration physical pass**,
3. the calibration residual relative to the population prediction,
4. a predefined personalization rule, and
5. a **different, independently acquired query pass** for evaluation.

The key calibration residual is

```text
ŷ_cal,pop = f(x_cal)
r_cal     = y_cal − ŷ_cal,pop
```

where `f(·)` is the frozen population predictor, `x_cal` is the calibration input, `y_cal` is the observed calibration outcome, and `r_cal` is the subject-specific calibration residual.

The query label is **never used during personalization**.

---

## Research Question

SAFE-Gait is designed around the following scientific question:

> **Does a single supervised calibration pass contain transferable subject-specific residual structure, and under what targets, directions, datasets, and individuals does that information improve or harm prediction?**

The project therefore emphasizes:

- cross-condition transfer,
- independent calibration and query physical passes,
- subject-level evaluation,
- external replication,
- transferability of residual structure,
- subject-level heterogeneity, and
- negative transfer.

---

## SAFE-Gait Workflow

```text
                         Unseen subject i
                         /             \
                        /               \
                       v                 v
            Calibration pass A       Query pass B
               supervised              unlabeled
                    |                     |
                    v                     v
          Population prediction   Population prediction
                    |                     |
                    v                     |
          Calibration residual             |
                    |                     |
                    v                     |
          One-shot personalization         |
          CP / SB / DS / MCS              |
                    \                     /
                     \                   /
                      v                 v
                  Personalized prediction
                           |
                           v
             Velocity / Step length / Step time
```

All population-level models, normalization statistics, residual priors, and method parameters are estimated from training subjects and frozen before test-subject personalization.

---

## Personalization Methods

SAFE-Gait currently compares four predefined methods:

| Abbreviation | Method |
|---|---|
| **CP** | Calibrated Population |
| **SB** | Simple Bias |
| **DS** | Diagonal Shrinkage |
| **MCS** | Multivariate Covariance Shrinkage |

Conceptually:

- **DS** uses target-wise / diagonal shrinkage.
- **MCS** uses full cross-target multivariate covariance.
- SAFE-Gait is the **overall personalization framework**; MCS is only one method within the framework.

> **Important:** This README intentionally does not provide simplified CP/SB/DS/MCS equations until their final mathematical definitions are checked line-by-line against the implementation. This avoids documenting formulas that are only conceptual approximations of the code.

---

## Prediction Targets

The predefined gait outcomes are:

- **Velocity**
- **Step length**
- **Step time**

The results consistently suggest that transferable residual structure is stronger for **velocity** and **step length**, while **step time** is weaker and less stable.

---

## Datasets

### WearGait-PD

WearGait-PD is the primary analysis dataset.

- 95 subjects
- 159 physical passes
- WPD021 excluded from the final analysis
- primary walking conditions: **SP** and **HP**
- primary transfer directions:
  - **SP → HP**
  - **HP → SP**
- pressure insole: Moticon OpenGo
- 16 pressure sensors per foot
- 32 pressure channels in total
- insole sampling rate: 100 Hz
- reference system: ProtoKinetics Zeno walkway
- 5-fold subject-level cross-validation
- training seeds: 7, 17, 27
- 225 completed formal models

### Insole–GAITRite

Insole–GAITRite is used as an independent replication dataset.

- 19 subjects
- 570 physical passes
- Slow: 195 passes
- Natural: 190 passes
- Fast: 185 passes
- P12_SP_7 excluded because of synchronization ambiguity
- 32 pressure sensors per foot
- 64 pressure channels in total
- insole sampling rate: 64 Hz
- reference system: GAITRite walkway
- primary transfer directions:
  - **Natural → Fast**
  - **Fast → Natural**
- secondary transfer directions:
  - Slow → Natural
  - Natural → Slow
  - Slow → Fast
  - Fast → Slow
- training seeds: 7, 17, 27
- **675 / 675** planned formal models completed
- failed runs: **0**

Development-only participants/settings are not used to redefine the primary confirmatory analysis.

---

## Frozen Evaluation Protocol

The evaluation protocol is deliberately strict to prevent information leakage and optimistic personalization estimates.

### Core rules

- Splits are performed at the **subject level**.
- Calibration and query observations must be **different physical passes**.
- Query labels are unavailable during personalization.
- Input normalization is estimated from **training subjects only**.
- Population residual statistics and personalization priors are estimated using **training-only out-of-fold residuals**.
- Model selection and population-level personalization quantities are frozen before final test-subject evaluation.
- The independent statistical unit is the **subject**.
- Training seeds are used only for robustness analysis and are **not treated as independent subjects**.
- The frozen protocol is not altered because an external replication result is unfavorable.
- No post-hoc threshold is introduced to improve GAITRite results.

Eligible subjects in the primary cross-condition analyses are:

| Dataset | Transfer direction | Eligible subjects |
|---|---:|---:|
| WearGait-PD | SP → HP | 23 |
| WearGait-PD | HP → SP | 23 |
| Insole–GAITRite | Natural → Fast | 18 |
| Insole–GAITRite | Fast → Natural | 18 |

---

## Evaluation Metrics

The principal prediction metric is **mean absolute error (MAE)**.

### Relative DS gain over CP

```text
100 × (MAE_CP − MAE_DS) / MAE_CP
```

A positive value indicates that DS improves over CP.

### Relative DS advantage over MCS

```text
100 × (MAE_MCS − MAE_DS) / MAE_MCS
```

A positive value indicates that DS outperforms MCS.

Confidence intervals are estimated using **subject-level bootstrap resampling**.

Subject-level personalization outcomes are additionally summarized as:

- **Benefit**
- **Unchanged**
- **Negative transfer**

The same frozen classification rules are used across analyses.

---

## Main Findings

The central finding is not that the most complex personalization model always wins.

Instead:

> **A single supervised calibration pass can contain subject-specific residual structure that transfers to another gait pass, but the strength and direction of this transfer depend on the prediction target, transfer direction, dataset, and individual subject.**

### WearGait-PD

The primary WearGait analysis shows relatively stable cross-condition personalization for:

- **Velocity**
- **Step length**

The effect for **step time** is weaker and less stable.

### External replication on GAITRite

Replication is **partial and direction dependent**.

In particular:

- Natural → Fast shows useful replication for velocity and step length.
- Fast → Natural is weaker.
- Fast → Natural velocity exhibits a direction reversal relative to the corresponding WearGait effect.
- Step time remains weak.
- Negative transfer occurs in a non-negligible subset of subjects.

### Complexity versus robustness

The results do not support the claim that full cross-target covariance is universally advantageous.

A simpler target-wise shrinkage strategy captures much of the transferable signal, suggesting that the dominant personalization structure is often **target specific** rather than strongly dependent on complex cross-target covariance.

---

## Cross-Dataset Summary

Representative DS gains relative to CP are:

| Dataset / transfer | Velocity (%) | Step length (%) | Step time (%) |
|---|---:|---:|---:|
| WearGait SP → HP | 27.646 | 35.613 | 6.049 |
| WearGait HP → SP | 26.469 | 32.335 | -0.436 |
| GAITRite Natural → Fast | 14.849 | 19.782 | -2.469 |
| GAITRite Fast → Natural | -6.354 | 8.450 | -1.320 |

These results motivate the paper's emphasis on **replication and limits**, rather than universal cross-dataset generalization.

---

## Subject-Level Heterogeneity

Average improvement does not imply that every individual benefits.

Across datasets and transfer directions, subjects can show:

- clear benefit,
- little or no change, or
- negative transfer.

This heterogeneity is a central part of SAFE-Gait rather than an analysis artifact.

The broader practical question is therefore:

> **When, and for whom, does one-shot personalization transfer?**

---

## Reproducibility Status

The main experimental pipeline has been frozen.

Completed items include:

- WearGait formal training runs
- GAITRite formal training runs
- frozen subject-level splits
- checkpoint replay checks
- hash consistency checks
- leakage audits
- bootstrap statistics
- transfer-coefficient analyses
- subject-level outcome analyses
- multi-seed robustness checks

Current scientific-development policy:

- do **not** retrain completed formal models unless a concrete implementation error is identified;
- do **not** tune a new method specifically to repair an unfavorable GAITRite result;
- do **not** treat seeds as additional independent samples;
- future extensions should preferably test the frozen framework on a **third independent gait dataset**.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/phr-devil/insole.git
cd insole
```

> **TODO — repository verification required:**  
> The exact Python version, dependency file, environment creation command, and package-installation command must be copied from the actual repository files before this section is finalized.

Examples such as `pip install -r requirements.txt` are intentionally **not** documented here until the corresponding file is verified to exist in the repository.

---

## Repository Structure

> **TODO — repository verification required.**

This section should be generated directly from the current GitHub tree rather than inferred from the manuscript or project handoff notes.

The final version should identify, at minimum:

- primary training entry point,
- evaluation entry point,
- dataset loaders,
- preprocessing utilities,
- configuration files,
- personalization implementation,
- statistical-analysis scripts,
- figure-generation scripts,
- checkpoints / generated artifacts policy, and
- tests or validation utilities.

---

## Data Preparation

The code uses WearGait-PD and Insole–GAITRite data.

> **TODO — repository and source verification required:**  
> Add the official dataset access URLs, licensing/access requirements, expected raw-data directory structure, preprocessing commands, and any files that cannot be redistributed with this repository.

Do not assume that raw or processed data are included in the GitHub repository unless explicitly confirmed.

---

## Running the Experiments

> **TODO — repository verification required.**

The final README should provide verified commands for at least:

1. WearGait SP → HP
2. WearGait HP → SP
3. GAITRite Natural → Fast
4. GAITRite Fast → Natural
5. one fixed seed
6. the complete three-seed experiment
7. statistical aggregation
8. figure/result generation

No executable command is included in this first version unless its script path and arguments have been checked against the actual repository.

---

## Expected Outputs

The final reproducibility documentation should identify the files/directories containing:

- prediction outputs,
- per-subject errors,
- CP / SB / DS / MCS comparisons,
- bootstrap confidence intervals,
- transfer coefficients,
- subject-level Benefit / Unchanged / Negative-transfer labels,
- seed-stability results, and
- manuscript figure data.

> **TODO — verify exact output paths from the repository.**

---

## Manuscript Figure Logic

The current paper is organized around the following figure sequence:

1. **Research question and sensing datasets**
2. **SAFE-Gait one-shot residual personalization framework**
3. **Experimental design and frozen evaluation protocol**
4. **Primary WearGait personalization performance**
5. **Transferable residual structure**
6. **Independent GAITRite replication**
7. **Cross-dataset replication and limits**
8. **Subject-level heterogeneity and negative transfer**

This reflects the overall scientific narrative:

```text
Problem
  ↓
Method
  ↓
Leakage-safe evaluation
  ↓
Primary evidence
  ↓
Mechanistic residual evidence
  ↓
External replication
  ↓
Replication limits
  ↓
Individual benefit and harm
```

---

## Project Status

SAFE-Gait is currently in the **manuscript-finalization and reproducibility-documentation stage**.

The main algorithmic and formal experimental work is considered frozen. Remaining work primarily concerns:

- final implementation-to-equation verification for CP / SB / DS / MCS,
- final repository documentation,
- final figures,
- manuscript writing,
- terminology consistency, and
- publication/citation information.

---

## Citation

A formal citation will be added after the manuscript is publicly available.

```bibtex
@article{safe_gait,
  title   = {SAFE-Gait: One-Shot Supervised Residual Personalization for Cross-Condition Gait Prediction},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```

---

## License

> **TODO — verify the repository license before publication.**

Do not add a license statement here unless a `LICENSE` file or an explicit licensing decision has been confirmed.

---

## Contact

For questions about the code or study, please use the repository's GitHub Issues page.

---

## Note on Documentation Accuracy

This README is intended to document the **frozen SAFE-Gait study**, not to retroactively modify the experimental protocol.

Where manuscript-level information is already frozen, it is reported directly. Where reproducibility details depend on the actual repository tree, configuration files, or command-line interfaces, this first version deliberately marks them as `TODO` rather than inventing paths, dependencies, formulas, or commands.
