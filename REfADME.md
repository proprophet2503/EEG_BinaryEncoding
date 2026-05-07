# EEG Binary Encoding — LH/RH Motor Classification

Binary classification of **Left Hand (LH) vs Right Hand (RH)** motor intention from EEG signals using a novel binary chain encoding feature representation.

Dataset: **EEGET-ALS** · Healthy participants only (id001–id170) · Scenarios 1 & 2

---

## Overview

This project investigates whether binary chain-encoded EEG amplitude sequences from motor cortex channels can discriminate left-hand from right-hand movement intention. Two paradigms are studied:

- **Motor imagery (Thinking)** — participant imagines the movement with eyes closed
- **Motor execution (Acting)** — participant physically performs the movement

The core hypothesis is that contralateral ERD patterns (C3/FC3 for RH, C4/FC4 for LH) are captured by the binary encoding chain's statistical features.

---

## Repository Structure

```
EEG_BinaryEncoding/
├── dataset/
│   ├── encoding chaining.csv       # Primary input (118,380+ rows)
│   ├── Steps.txt                   # Pipeline workflow notes
│   ├── Train samples 118380.txt    # Sample count metadata
│   └── paper_complete.tex          # LaTeX paper skeleton
│
├── classifier_notebook/
│   ├── EDA.ipynb                   # Exploratory data analysis
│   ├── preprocessing.ipynb         # Feature engineering → wide CSV
│   ├── modelling.ipynb             # ML training & evaluation
│   ├── eegnet-*.ipynb              # Deep learning (EEGNet) classifiers
│   ├── 01–08_*.ipynb               # Systematic experiment notebooks
│   ├── COPILOT_INSTRUCTIONS.md     # Full pipeline specification
│   │
│   ├── eda_outputs/                # EDA visualizations & stats
│   ├── wide_model_outputs/         # Cross-subject wide-feature results
│   ├── wide_subjectdep_outputs/    # Subject-dependent wide results
│   ├── erd_ers_model_outputs/      # Cross-subject ERD/ERS results
│   ├── erd_ers_subjectdep_outputs/ # Subject-dependent ERD/ERS results
│   ├── binary_wide_S4_vs_S7_outputs/
│   ├── binary_erd_ers_S5_vs_S7_outputs/
│   ├── binary_erd_ers_preprocessingModified_*/
│   ├── binary_erd_ers_all_pairs_outputs/
│   └── results_local_cpu_gpu/      # Full experiment results
│       ├── preprocessing/
│       ├── eda/
│       ├── modelling/
│       ├── deep_learning/
│       ├── deep_learning_edf_lh_rh/
│       └── saved_models/           # .pkl / .pt / .joblib model files
│
└── features.csv/                   # Intermediate feature CSVs
```

---

## Dataset

### Source
**EEGET-ALS Dataset** — multi-scenario EEG with both healthy participants and ALS patients.  
This project uses **healthy subjects only** (`id001`–`id170`).

### Input CSV: `encoding chaining.csv`

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | str | `id1`–`id170` (healthy); `ALS01`–`ALS06` (excluded) |
| `scenario_id` | int | 1 = Lift Left Hand, 2 = Lift Right Hand |
| `task` | str | `Thinking`, `Acting`, `Typing`, `Resting` |
| `channel` | str | EEG channel (e.g. `C3`, `C4`, `Cz`) |
| `subband` | str | `Delta`, `Theta`, `Alpha`, `Beta`, `Gamma` |
| `feature` | str | Feature type (e.g. `mav`) |
| `chain_sequence` | str | Binary string encoding amplitude direction |
| `chain_ratio` | float | Proportion of `1`s in chain_sequence |

**Classification target:**
- `scenario_id == 1` → label `0` (LH)
- `scenario_id == 2` → label `1` (RH)

### Filtering Criteria

| Filter | Values Kept |
|--------|-------------|
| Subject type | Healthy (`id` prefix) |
| Scenarios | 1 (LH), 2 (RH) |
| Tasks | `Thinking`, `Acting` |
| Channels | `C3`, `Cz`, `C4`, `FC3`, `FC4`, `CP3`, `CP4` |
| Subbands | `Alpha`, `Beta`, `Gamma` |

---

## Feature Representation

### Binary Chain Encoding
Each EEG window sequence is encoded as a binary string where `1` = amplitude increase and `0` = amplitude decrease between consecutive time points.

### Engineered Features (from `chain_sequence`)

| Feature | Description |
|---------|-------------|
| `chain_len` | Total sequence length |
| `chain_ones` / `chain_zeros` | Count of 1s and 0s |
| `chain_ones_ratio` | Proportion of 1s (≈ `chain_ratio`) |
| `chain_longest_run1` | Longest consecutive run of 1s |
| `chain_longest_run0` | Longest consecutive run of 0s |
| `chain_transitions` | Number of 0↔1 transitions |
| `chain_entropy` | Binary entropy: $-p_1\log_2 p_1 - p_0\log_2 p_0$ |

Features are pivoted to **wide format** — one row per sample, columns named `[channel]_[subband]_[feature]`.

---

## Neuroscience Rationale

### Motor Cortex Lateralization
- **Scenario 1 (LH):** Expected ERD in right motor cortex → C4, FC4
- **Scenario 2 (RH):** Expected ERD in left motor cortex → C3, FC3
- **Midline Cz:** Bilateral activation indicator

### Motor-Relevant Frequency Bands
| Band | Range | Role |
|------|-------|------|
| Alpha | 8–13 Hz | ERD during motor tasks (power decrease = activation) |
| Beta | 13–30 Hz | Strong ERD during movement, ERS after cessation |
| Gamma | 30–50 Hz | High-frequency activity during execution |

---

## Pipeline

```
encoding chaining.csv
        │
        ▼
   EDA.ipynb              ← distribution analysis, channel/subband stats,
        │                    Mann-Whitney U significance, lateralization plots
        ▼
preprocessing.ipynb       ← filter → chain feature engineering → wide pivot
        │                    → features_lh_rh.csv
        ▼
modelling.ipynb           ← Stratified K-Fold CV, LOSO, hold-out evaluation
        │                    → cv_results.csv, final_metrics.csv, saved models
        ▼
eegnet-*.ipynb            ← deep learning on raw EDF (EEGNet architecture)
```

---

## Models

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline linear classifier |
| Random Forest | 200 trees, max_depth=10 |
| SVM (RBF) | C=1.0, gamma='scale' |
| XGBoost | 200 estimators, max_depth=5 (optional) |
| LightGBM | 200 estimators, max_depth=5 (optional) |
| EEGNet | Deep learning on raw EDF signals |

**Preprocessing pipeline:** `RobustScaler` → `SelectKBest` (top 50 features by F-score) → classifier

---

## Evaluation Protocol

| Protocol | Description |
|----------|-------------|
| Stratified K-Fold (5-fold) | Cross-validation preserving class balance |
| LOSO (Leave-One-Subject-Out) | Cross-subject generalization |
| Hold-out (80/20) | Final evaluation on stratified split |

**Metrics:** Accuracy, Balanced Accuracy (BAC), F1, ROC-AUC, Cohen's Kappa

> Cross-subject LOSO accuracy is typically lower than within-subject. Report **Balanced Accuracy** and **ROC-AUC** as primary metrics due to potential class imbalance.

---

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `features_lh_rh.csv` | `classifier_notebook/` | Wide-format features, one row per sample |
| `cv_results.csv` | `results_local_cpu_gpu/modelling/` | 5-fold CV scores per model |
| `loso_results.csv` | `results_local_cpu_gpu/modelling/` | LOSO per-subject breakdown |
| `final_metrics.csv` | `results_local_cpu_gpu/modelling/` | Hold-out metrics, all models |
| `feature_importance.csv` | `results_local_cpu_gpu/modelling/` | Top features by model importance |
| `confusion_matrix.png` | `results_local_cpu_gpu/modelling/` | Raw counts + normalized |
| `roc_curves.png` | `results_local_cpu_gpu/modelling/` | All models on one plot |
| `importance_heatmap.png` | `results_local_cpu_gpu/modelling/` | Channel × Subband importance |
| `*.pkl` / `*.joblib` | `results_local_cpu_gpu/saved_models/` | Serialized best-model objects |

---

## Experiment Variants

| Notebook | Feature Type | Protocol |
|----------|-------------|----------|
| `02_CrossSubject_Wide.ipynb` | Wide chain features | Cross-subject |
| `03_CrossSubject_ERD_ERS.ipynb` | ERD/ERS band power | Cross-subject |
| `04_CrossSubject_Encoding.ipynb` | Binary encoding | Cross-subject |
| `05_SubjectDependent_Wide.ipynb` | Wide chain features | Subject-dependent |
| `06_SubjectDependent_ERD_ERS.ipynb` | ERD/ERS band power | Subject-dependent |
| `07_Binary_Wide.ipynb` | Wide features | Binary LH/RH |
| `08_Binary_ERD_ERS*.ipynb` | ERD/ERS (variants) | Binary LH/RH |
| `eegnet-*.ipynb` | Raw EDF signals | Deep learning |

---

## Requirements

```
python >= 3.8
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
joblib
xgboost        # optional
lightgbm       # optional
mne            # for EDF loading (EEGNet notebooks)
torch          # for EEGNet
```

Install:
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn joblib xgboost lightgbm mne torch
```

---

## Quick Start

```python
# 1. Run EDA
#    Open classifier_notebook/EDA.ipynb → Run All

# 2. Generate features
#    Open classifier_notebook/preprocessing.ipynb → Run All
#    Output: classifier_notebook/features_lh_rh.csv

# 3. Train and evaluate models
#    Open classifier_notebook/modelling.ipynb → Run All
#    Output: results/ directory with CSVs and plots

# 4. (Optional) Deep learning
#    Open classifier_notebook/eegnet-*.ipynb → Run All
```

> **Paths:** Update the configuration cells in each notebook to match your local directory structure before running.

---

## Known Issues

| Issue | Solution |
|-------|----------|
| `chain_sequence` is object type | Convert with `str(val)` before parsing |
| Missing columns after pivot | Some channel/subband combos absent → fill NaN with median |
| `SelectKBest k > n_features` | Uses `min(N_SELECT_K, X.shape[1])` automatically |
| LightGBM warning spam | Pass `verbose=-1` to constructor |
| Low LOSO accuracy | Expected for cross-subject EEG; use BAC not raw accuracy |

---

## References

- EEGET-ALS Dataset — healthy participant EEG recordings, scenarios 1–9
- EEGNet: Lawhern et al. (2018). *EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces*
- ERD/ERS: Pfurtscheller & Lopes da Silva (1999). *Event-related EEG/MEG synchronization and desynchronization*
