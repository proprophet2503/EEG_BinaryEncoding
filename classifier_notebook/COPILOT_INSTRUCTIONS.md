# 🧠 Copilot Instructions — EEG LH/RH Classifier Notebooks

> **Project:** Binary classification of Left Hand (LH) vs Right Hand (RH)  
> from EEGET-ALS EEG data of **healthy participants only** (id001–id170).  
> **Scenarios:** 1 (Lift Left Hand) & 2 (Lift Right Hand).

---

## 📁 Paths Reference

| Variable | Path |
|---|---|
| `ENCODING_CSV` | `/home/jeremy-mboe/Documents/Kuliah/Sem4/EEG_ALS/WEB/dataset/encoding chaining.csv` |
| `PROCESSING_DIR` | `/home/jeremy-mboe/Documents/Kuliah/Sem4/EEG_ALS/WEB/EEG_web/processing/` |
| `DATASET_ROOT` | `/home/jeremy-mboe/Documents/Kuliah/Sem4/EEG_ALS/WEB/dataset/EEGET-ALS Dataset/dataset` |
| `NOTEBOOK_DIR` | `/home/jeremy-mboe/Documents/Kuliah/Sem4/EEG_ALS/WEB/classifier_notebook` |

---

## 📄 Input CSV Schema

File: `encoding chaining.csv`

| Column | Type | Description |
|---|---|---|
| `subject_id` | str | `id1`–`id170` for healthy; `ALS01`–`ALS06` for ALS |
| `scenario` | str | Scenario name (e.g. `scenario1`) |
| `scenario_id` | int | 1–9 |
| `filename` | str | Full path to source EDF |
| `task` | str | `Thinking`, `Acting`, `Typing`, `Resting` |
| `channel` | str | EEG channel name (e.g. `C3`, `Cz`) |
| `subband` | str | `Delta`, `Theta`, `Alpha`, `Beta`, `Gamma` |
| `feature` | str | Feature type (e.g. `mav`) |
| `chain_sequence` | str | Binary string (e.g. `01001111...`) |
| `chain_ratio` | float | Ratio of 1s in chain_sequence |

**Target classes:**
- `scenario_id == 1` → label `0` (LH = Lift Left Hand)
- `scenario_id == 2` → label `1` (RH = Lift Right Hand)

---

## 📓 Notebook 1 — `EDA.ipynb`

### Purpose
Explore the raw encoding CSV to understand the data before any modelling.

### Instructions for Copilot

```
Create a Jupyter notebook EDA.ipynb in:
/home/jeremy-mboe/Documents/Kuliah/Sem4/EEG_ALS/WEB/classifier_notebook/

The notebook must cover:

1. IMPORTS
   - numpy, pandas, matplotlib, seaborn, scipy.stats
   - warnings.filterwarnings('ignore')

2. LOAD DATA
   - Load: /home/jeremy-mboe/Documents/Kuliah/Sem4/EEG_ALS/WEB/dataset/encoding chaining.csv
   - Print shape, dtypes, head(3)

3. DATASET INVENTORY
   - Count unique: subjects, scenarios, tasks, channels, subbands
   - Show value_counts for each categorical column
   - Identify ALS vs healthy subjects (ALS prefix vs id prefix)

4. FILTER TO STUDY SCOPE
   - Keep only healthy subjects (subject_id starts with 'id')
   - Keep only scenario_id in [1, 2]
   - Keep only tasks: ['Thinking', 'Acting']
   - Report rows before and after filter

5. CLASS BALANCE
   - Bar chart: LH vs RH sample counts
   - Bar chart: LH vs RH per task (Thinking, Acting)
   - Report imbalance ratio

6. CHAIN SEQUENCE ANALYSIS
   - Plot distribution of chain_ratio for LH vs RH (overlapping histogram + KDE)
   - Plot distribution of chain sequence length per subband
   - Calculate mean chain_ratio per (class, channel, subband)
   - Heatmap: mean chain_ratio — Channel × Subband for LH
   - Heatmap: mean chain_ratio — Channel × Subband for RH
   - Difference heatmap: RH - LH chain_ratio

7. FEATURE DISTRIBUTIONS
   - For features that might appear as extra numeric columns
   - Box plots: chain_ratio distribution per channel (filter to motor channels: C3, Cz, C4)
   - Box plots: chain_ratio distribution per subband

8. MOTOR CORTEX FOCUS
   - Filter to: channels = ['C3', 'Cz', 'C4', 'FC3', 'FC4']
              subbands = ['Alpha', 'Beta', 'Gamma']
   - Violin plot: chain_ratio for LH vs RH per channel
   - Violin plot: chain_ratio for LH vs RH per subband
   - Statistical test (Mann-Whitney U): LH vs RH chain_ratio for each (channel, subband)
   - Show significant pairs (p < 0.05)

9. CORRELATION ANALYSIS
   - Point-biserial correlation of chain_ratio vs label (0/1) per (channel, subband)
   - Heatmap: correlation values — Channel × Subband
   - Identify top 10 most discriminative (channel, subband) pairs

10. SUBJECT-LEVEL ANALYSIS
    - Per-subject mean chain_ratio for LH and RH
    - Scatter plot: subject mean LH chain_ratio vs RH chain_ratio
    - Check if subjects show consistent lateralization pattern

11. TASK COMPARISON
    - Compare chain_ratio for Thinking vs Acting tasks per class
    - Are there differences in discriminability between tasks?

12. SAVE EDA SUMMARY
    - Save: eda_summary.csv with key statistics
    - Save all plots as PNG in NOTEBOOK_DIR
```

---

## 📓 Notebook 2 — `preprocessing.ipynb`

### Purpose
Load encoding CSV → filter → engineer features → pivot to wide format → save `features_lh_rh.csv`.

### Instructions for Copilot

```
Create a Jupyter notebook preprocessing.ipynb in:
/home/jeremy-mboe/Documents/Kuliah/Sem4/EEG_ALS/WEB/classifier_notebook/

The notebook must:

1. IMPORTS
   - numpy, pandas, matplotlib, seaborn, scipy.stats
   - from pathlib import Path

2. CONFIGURATION CELL (all paths and params as variables)
   ENCODING_CSV  = '/home/jeremy-mboe/.../encoding chaining.csv'
   OUTPUT_DIR    = '/home/jeremy-mboe/.../classifier_notebook'
   OUTPUT_CSV    = OUTPUT_DIR + '/features_lh_rh.csv'
   TARGET_SCENARIOS = [1, 2]           # LH, RH only
   TARGET_TASKS  = ['Thinking', 'Acting']
   MOTOR_CHANNELS = ['C3', 'Cz', 'C4', 'FC3', 'FC4', 'CP3', 'CP4']
   TARGET_SUBBANDS = ['Alpha', 'Beta', 'Gamma']

3. LOAD CSV + INVENTORY
   - Load encoding chaining.csv
   - Print shape, unique values per column

4. FILTER
   - Healthy subjects only: subject_id.startswith('id')
   - scenario_id in [1, 2]
   - task in ['Thinking', 'Acting']
   - channel in MOTOR_CHANNELS
   - subband in TARGET_SUBBANDS
   - Print shape at each step

5. FEATURE ENGINEERING from chain_sequence
   Implement function chain_features(chain_seq: str) -> dict that returns:
   - chain_len: length of sequence
   - chain_ones: count of 1s
   - chain_zeros: count of 0s
   - chain_ones_ratio: ones / len (same as chain_ratio but computed)
   - chain_longest_run1: max consecutive 1s
   - chain_longest_run0: max consecutive 0s
   - chain_transitions: number of 0↔1 transitions
   - chain_entropy: binary entropy -(p1*log2(p1) + p0*log2(p0))
   Apply to all rows, concat as new columns.

6. BUILD LABELS
   - label = 0 if scenario_id == 1 (LH), 1 if scenario_id == 2 (RH)
   - label_name = 'LH' or 'RH'

7. IDENTIFY NUMERIC FEATURES
   - All numeric cols excluding meta columns
   - Meta = {subject_id, scenario, scenario_id, filename, task, channel,
             subband, feature, chain_sequence, label, label_name}

8. PIVOT TO WIDE FORMAT
   Group key = [subject_id, scenario_id, filename, task, label, label_name]
   For each numeric feature:
     pivot_table(index=GROUP_KEY, columns='channel'+'_'+'subband'+'_'+feat, values=feat)
   Concatenate all pivots → 1 row = 1 sample
   Remove duplicate columns.

9. MISSING VALUES
   - Compute missing % per column
   - Drop columns >50% missing
   - Fill remaining NaN with column median
   - Show missing value heatmap (first 60 features)

10. CLASS BALANCE CHECK
    - Bar chart: overall LH vs RH
    - Bar chart: per task (Thinking, Acting)

11. FEATURE STATISTICS
    - Describe numeric features
    - Add coefficient of variation (CV = std/mean)
    - Show top 10 by std

12. CORRELATION WITH LABEL
    - Point-biserial correlation for each feature vs label
    - Plot horizontal bar chart of top 20 features by |correlation|

13. SAVE
    - wide_df.to_csv(OUTPUT_CSV, index=False)
    - Save preprocessing_summary.csv: shape, class counts, n_subjects
    - Print final shape and class counts
    - Save all plots as PNG in OUTPUT_DIR

OUTPUT: features_lh_rh.csv
Columns: subject_id, scenario_id, filename, task, label, label_name,
         + [channel]_[subband]_[feature] columns (wide format)
```

---

## 📓 Notebook 3 — `modelling.ipynb`

### Purpose
Load `features_lh_rh.csv` → train multiple classifiers → evaluate → save results.

### Instructions for Copilot

```
Create a Jupyter notebook modelling.ipynb in:
/home/jeremy-mboe/Documents/Kuliah/Sem4/EEG_ALS/WEB/classifier_notebook/

The notebook must:

1. IMPORTS
   - numpy, pandas, matplotlib, seaborn, joblib, warnings
   - from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, cross_validate, train_test_split
   - from sklearn.pipeline import Pipeline
   - from sklearn.preprocessing import RobustScaler
   - from sklearn.feature_selection import SelectKBest, f_classif
   - from sklearn.linear_model import LogisticRegression
   - from sklearn.ensemble import RandomForestClassifier
   - from sklearn.svm import SVC
   - from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
       roc_auc_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay,
       classification_report, roc_curve)
   - Try-import XGBoost and LightGBM with HAS_XGB / HAS_LGB flags
   - from scipy.stats import pointbiserialr

2. CONFIG
   - BASE_DIR = '.../classifier_notebook'
   - INPUT_CSV = BASE_DIR + '/features_lh_rh.csv'
   - RESULTS_DIR = BASE_DIR + '/results'
   - MODELS_DIR  = BASE_DIR + '/saved_models'
   - os.makedirs for both dirs
   - RANDOM_STATE = 42
   - N_FOLDS = 5
   - N_SELECT_K = 50

3. LOAD DATA
   - Read features_lh_rh.csv
   - META_COLS = {subject_id, scenario_id, scenario, filename, task, label, label_name}
   - feature_cols = all non-meta columns
   - X = df[feature_cols].values.astype(float)
   - y = df['label'].values
   - groups = df['subject_id'].values

4. DEFINE PIPELINES
   Make function make_pipeline(clf, n_features=N_SELECT_K):
     return Pipeline([
       ('scaler', RobustScaler()),
       ('select', SelectKBest(f_classif, k=min(n_features, X.shape[1]))),
       ('clf', clf)
     ])
   
   MODELS dict:
   - 'Logistic Regression': LR(C=1.0, max_iter=1000)
   - 'Random Forest': RF(n_estimators=200, max_depth=10)
   - 'SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
   - 'XGBoost' (if available): XGBClassifier(n_estimators=200, max_depth=5)
   - 'LightGBM' (if available): LGBMClassifier(n_estimators=200, max_depth=5)

5. STRATIFIED K-FOLD CV
   - StratifiedKFold(n_splits=5, shuffle=True)
   - SCORING = ['accuracy', 'balanced_accuracy', 'f1', 'roc_auc']
   - cross_validate each model
   - Print mean ± std for each metric per model

6. CV RESULTS TABLE
   - DataFrame with columns: Model, Accuracy, Bal. Accuracy, F1, ROC-AUC
   - Format as "mean ± std"
   - Sort by BAC_mean descending
   - Save cv_results.csv

7. MODEL COMPARISON BAR CHART
   - 4 subplots (one per metric), bars per model with error bars
   - Show value labels on bars
   - Save model_comparison.png

8. LEAVE-ONE-SUBJECT-OUT (LOSO)
   - Use best model from CV
   - LeaveOneGroupOut() with groups=subject_id
   - cross_validate on best model
   - Print mean ± std
   - Per-subject bar chart (Accuracy, BAC, AUC per subject)
   - Save loso_results.csv and loso_per_subject.png

9. HOLD-OUT EVALUATION (80/20 stratified split)
   - X_train, X_test, y_train, y_test = train_test_split(stratify=y)
   - Fit best model
   - classification_report(target_names=['LH','RH'])
   - Print Cohen's Kappa and ROC-AUC

10. CONFUSION MATRIX
    - 2 subplots: raw counts + normalized
    - Use ConfusionMatrixDisplay
    - Save confusion_matrix.png

11. ROC CURVES — ALL MODELS
    - One plot, all models on same axes
    - Label each curve with model name and AUC
    - Add random diagonal line
    - Save roc_curves.png

12. FEATURE IMPORTANCE
    - For tree-based: feature_importances_
    - For LR/SVM: |coef_|
    - Horizontal bar chart of top 30 features
    - Save feature_importance.png
    - Save feature_importance.csv

13. IMPORTANCE HEATMAP (Channel × Subband)
    - Parse feature names: channel_subband_featuretype
    - Sum importance per (channel, subband)
    - Seaborn heatmap with annotations
    - Save importance_heatmap.png

14. FINAL METRICS TABLE (all models on hold-out)
    - Columns: Model, Accuracy, Balanced_Acc, F1, ROC_AUC, Cohen_Kappa, CV_BAC_mean, CV_AUC_mean
    - Sort by Balanced_Acc descending
    - Heatmap visualization (RdYlGn colormap, vmin=0.3, vmax=1.0)
    - Save final_metrics.csv and metrics_heatmap.png

15. SAVE BEST MODEL
    - Fit best model on full dataset (all X, y)
    - joblib.dump to MODELS_DIR/[ModelName]_best.pkl
    - Save feature_cols.txt (one feature name per line)
    - Print list of all saved files with sizes

SAVED FILES in results/:
  cv_results.csv, model_comparison.png, loso_results.csv,
  loso_per_subject.png, confusion_matrix.png, roc_curves.png,
  feature_importance.csv, feature_importance.png,
  importance_heatmap.png, final_metrics.csv, metrics_heatmap.png
```

---

## 🔧 Processing Modules Available

The following modules are in `/home/jeremy-mboe/Documents/Kuliah/Sem4/EEG_ALS/WEB/EEG_web/processing/`:

| Module | Key class/function | Purpose |
|---|---|---|
| `loader.py` | `EEGLoader` | Load EDF files, extract task segments, detect metadata |
| `features.py` | `EEGFeatures` | Band power, MAV, variance, std, ERD/ERS per channel/subband |
| `epoching.py` | `EpochEngine` | Fixed-length epochs, sliding windows, feature aggregation |
| `filters.py` | `EEGFilters` | Bandpass, notch, ICA, bad channel detection, CAR |
| `psd.py` | `PSDAnalyzer` | Welch/Multitaper PSD, band power extraction |
| `encoding.py` | `encode_dataset`, `encode_single_edf` | Batch encode EDF → feature CSV |
| `connectivity.py` | `ConnectivityAnalyzer` | PLI/wPLI functional connectivity |
| `delta.py` | `DeltaCalculator` | Delta between tasks, group transition analysis |
| `statistics.py` | `StatisticalTests` | Mann-Whitney, t-test, Cohen's d, FDR correction |

**To use in notebooks:**
```python
import sys
sys.path.insert(0, '/home/jeremy-mboe/Documents/Kuliah/Sem4/EEG_ALS/WEB/EEG_web')
from processing.loader import EEGLoader
from processing.features import EEGFeatures
from processing.epoching import EpochEngine
```

---

## 📊 Key Neuroscience Context

### Why Scenarios 1 & 2?
- **Scenario 1** = Lift Left Hand → expected ERD in **right motor cortex** (C4, FC4)
- **Scenario 2** = Lift Right Hand → expected ERD in **left motor cortex** (C3, FC3)
- The `chain_sequence` binary string captures whether the signal amplitude is increasing (1) or decreasing (0) between adjacent windows.

### Task Types
| Task annotation | Code | Description |
|---|---|---|
| `Thinking` | task i | Motor imagery (closed eyes, imagine movement) |
| `Acting` | task ii | Physical movement execution (open eyes) |
| `Typing` | task iii | Eye-tracking spelling (has simultaneous ET data) |
| `Resting` | — | Rest between tasks |

### Motor-relevant channels
- **Contralateral to left hand:** C4, FC4, CP4 (right hemisphere)
- **Contralateral to right hand:** C3, FC3, CP3 (left hemisphere)
- **Midline:** Cz (bilateral activation)

### Motor-relevant subbands
- **Alpha (8–13 Hz):** ERD during motor tasks (power decrease = activation)
- **Beta (13–30 Hz):** Strong ERD during movement, ERS after movement ends
- **Gamma (30–50 Hz):** High-frequency activity during movement execution

---

## ⚠️ Common Issues & Solutions

| Issue | Solution |
|---|---|
| `chain_sequence` column is object type | Convert with `str(val)` before parsing |
| Missing feature columns after pivot | Some channel/subband combos absent → fill NaN with median |
| Very low LOSO accuracy | Normal for cross-subject EEG; report BAC not raw accuracy |
| Imbalanced classes | Use `balanced_accuracy_score`, `f1_score`, `class_weight='balanced'` |
| SelectKBest k > n_features | Use `min(N_SELECT_K, X.shape[1])` |
| LightGBM warning spam | Pass `verbose=-1` to constructor |

---

## ✅ Expected Outputs

```
classifier_notebook/
├── features_lh_rh.csv            ← output of preprocessing.ipynb
├── preprocessing_summary.csv
├── missing_heatmap.png
├── class_balance.png
├── feature_correlation.png
├── results/
│   ├── cv_results.csv
│   ├── model_comparison.png
│   ├── loso_results.csv
│   ├── loso_per_subject.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   ├── importance_heatmap.png
│   ├── final_metrics.csv
│   └── metrics_heatmap.png
└── saved_models/
    ├── [BestModel]_best.pkl
    └── feature_cols.txt
```
