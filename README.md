# PPG-Based Blood Pressure Estimation  
End-to-End Signal Processing, Feature Engineering, Machine Learning, and Clinical Validation

This repository contains a complete machine learning pipeline for estimating systolic and diastolic blood pressure (SBP/DBP) from photoplethysmography (PPG) waveforms. The project integrates physiological signal processing, handcrafted feature extraction, model training using multiple regressors, and clinical validation on a real ambulatory blood pressure monitoring (ABPM) dataset.

---

## Project Structure

```
.
├── extract_features_and_predict_bp_from_ppg_eval.ipynb   # Core ML pipeline
├── KolkataStatsTests.ipynb                               # Clinical validation and statistical analysis
├── Validation Report.pdf                                 # Summary of best results and evaluation
├── features/                                             # Auto-generated feature CSVs
├── results/                                              # Model performance outputs
└── README.md
```

---

## Overview

This project implements an end-to-end framework for cuff-less blood pressure estimation using PPG. It includes:

1. Signal preprocessing and cardiac cycle segmentation  
2. Morphological feature extraction from each cycle  
3. Aggregation and correlation-based feature pruning  
4. Training multiple machine learning models  
5. Subject-wise cross-validation  
6. Clinical validation using ABPM data from a Kolkata study  

The pipeline is designed to follow realistic physiological and clinical validation practices.

---

## 1. Signal Processing Pipeline

### Cycle Detection
- Uses `scipy.signal.find_peaks` to identify cycle start points  
- Computes a template cycle from median cycle length  
- Evaluates cycle quality using two signal quality indices (SQI):  
  - Pearson correlation with the template  
  - Correlation with a resampled version of each cycle  
- Only cycles with SQI ≥ 0.8 are kept

### Cycle Cleaning
- Time-interpolation to align cycle timestamps  
- Baseline normalization  
- Outlier trimming based on the 80th percentile threshold

These steps ensure high-quality and morphologically consistent cardiac cycles for feature extraction.

---

## 2. Morphological Feature Extraction

For each valid cardiac cycle, the pipeline extracts:

- Systolic peak timestamp  
- Systolic rise time (T_S)  
- Diastolic decay time (T_D)  
- Widths at fractional amplitudes (10%, 25%, 33%, 50%, 66%, 75%)  
- Systolic-to-diastolic timing ratios  
- Cycle period (CP)

Features are aggregated over each window (e.g., 30 seconds) using:

- Mean  
- Variance  

The result is a structured and interpretable feature set representing PPG morphology.

---

## 3. Machine Learning Pipeline

### Feature Pruning
Features with correlation greater than 0.95 are removed.

### Models Implemented
- ElasticNet Regression  
- Gradient Boosting Regressor  
- Random Forest Regressor  
- LightGBM Regressor  
- Dummy Regressor (mean baseline)

### Cross-Validation Strategy
Uses leave-k-subjects-out cross-validation with k ∈ {1, 2, 3}, providing a realistic estimate of generalization across unseen individuals.

Performance metrics include:

- MAE (Mean Absolute Error)  
- RMSE (Root Mean Squared Error)  
- MAPE (Mean Absolute Percentage Error)  
- Standard deviations of metrics across folds  

Results are saved in timestamped CSV files under the `results/` directory.

---

## 4. ML Evaluation Results (AIME 2020 EVAL Dataset)

For each predicted variable (SBP, DBP) and each value of k, the notebook computes and stores:

- Mean and standard deviation of MAE  
- Mean and standard deviation of RMSE  
- Mean and standard deviation of MAPE  
- Best-performing configurations

Final summaries are exported to:

```
results/YYYY-MM-DD_best_results_eval.csv
```

---

## 5. Clinical Validation (Kolkata ABPM Dataset)

Implemented in `KolkataStatsTests.ipynb`, this component provides a detailed clinical evaluation using a real ABPM dataset.

### Bland–Altman Analysis
- Per-subject and overall agreement plots  
- Mean bias, limits of agreement, and outlier counts  
- Exported as PNGs for each subject  

### Dipping Analysis
- Identification of nighttime dipping interval  
- Linear regression of BP decay (reference and predicted)  
- Comparison of dipping slopes  
- Summary tables of dipping strength  

### Error Distribution and Normality
For each subject:
- Mean error  
- Mean absolute error  
- RMSE  
- Mean absolute percentage error  
- Percent of data within 1/2/3 standard deviations  

### Demographic Correlation
Using a subject metadata file, the notebook examines trends across:
- Age  
- Gender  
- Medical group  
- HbA1C levels  
- Chronic conditions  
- Medication regimens  

This section connects ML error patterns to clinically relevant subgroups.

---

## 6. Data Sources

### AIME 2020 EVAL Dataset
Processed 30-second PPG windows (normalized, back-filled).
Citation:  
Morassi Sasso, Ariane (2020). Processed EVAL Dataset. https://doi.org/10.6084/m9.figshare.12649691

### Kolkata ABPM Dataset
Real ambulatory BP monitoring dataset from a multisubject study used for clinical validation.

---

## 7. Reproducing the Pipeline

### Extract Features
```python
path = extract_features(
    csv=True,
    time_delta='30 seconds',
    time_delta_type='bfill',
    experiment_type='eval',
    special_filter='norm',
    verbose=False
)
```

### Train Models
```python
df = pd.read_csv(path)
results = predict_bp_from_ppg(df, predicted_variable='SBP', k=2)
```

### Clinical Validation
```python
generateBlandAltmanGraphs(1, 30)
generateDippingGraphStats(False)
generateMeanStats('by subject', False)
```

---

## 8. Strengths of This Project

- Combines signal processing, ML, and clinical validation in one pipeline  
- Produces interpretable, physiologically grounded features  
- Uses subject-based cross-validation for more realistic evaluation  
- Includes complete clinical-grade analysis (Bland–Altman, dipping, circadian trends)  
- Demonstrates a multi-disciplinary understanding of biosignals, modeling, and statistics  

---

## 9. Validation Report

A full summary of best-performing models, error characteristics, and clinical agreement is provided in `Validation Report.pdf`.

---

## 10. Future Work

- Incorporating second-derivative PPG features  
- Adding neural time-series models (TCN, Transformer)  
- Applying motion artifact filtering  
- Personalized BP models  
- Testing on additional datasets (e.g., MIMIC-III waveform database)

---

## Contributors

**Sriram Bhimaraju**  
Signal Processing, Machine Learning, and Clinical Analytics

