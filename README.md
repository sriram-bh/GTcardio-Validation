PPG-Based Blood Pressure Estimation
End-to-End Signal Processing, Feature Engineering, Machine Learning, and Clinical Validation

This repository contains a complete machine-learning pipeline for estimating systolic and diastolic blood pressure (SBP/DBP) from Photoplethysmography (PPG).
It includes:

Physiological signal processing and cycle segmentation

Handcrafted morphological feature extraction

Feature selection via correlation-based pruning

Multi-model ML regression with subject-wise cross-validation

A full suite of clinical validation analyses

A secondary dataset study on ABPM data from Kolkata

This project re-implements and extends the AIME 2020 blood pressure pipeline and adds a complete downstream clinical statistics workflow.

Project Structure
.
├── extract_features_and_predict_bp_from_ppg_eval.ipynb   # Core ML pipeline
├── KolkataStatsTests.ipynb                               # Clinical validation
├── Validation Report.pdf                                 # Summary of results
├── features/                                             # Generated feature CSVs
├── results/                                              # Model performance outputs
└── README.md

Overview

This project implements a signal-processing + ML framework to estimate BP from raw PPG waveforms. The workflow includes:

PPG preprocessing

High-quality cardiac-cycle detection

Cycle-level morphological feature extraction

Aggregated window statistics (mean/variance)

Correlation-based feature pruning

Model training and evaluation

Clinical-grade validation on ABPM datasets

The pipeline is robust, interpretable, and designed to mimic FDA-style physiological validation.

1. Signal Processing Pipeline
Cycle Detection

Uses scipy.signal.find_peaks to detect local minima that correspond to cycle starts

Builds a template cycle from median cycle length

Computes:

Pearson correlation between each cycle and the template

A resampled correlation score

Cycles with SQI < 0.8 are rejected.

Cycle Cleaning

Performs:

Interpolation to aligned timestamps

Baseline normalization

Outlier trimming based on 80th percentile thresholds

Why This Matters

Clean cycle-level segmentation dramatically improves the interpretability and stability of BP estimation from PPG—a known hard problem due to motion artifacts and morphological variability.

2. Morphological Feature Extraction

For each valid cardiac cycle, the algorithm computes:

Systolic Peak Timing (T_S)

Diastolic duration (T_D)

Fractional amplitude widths (DW_10, DW_25, DW_33, DW_50, DW_66, DW_75)

Systolic–Diastolic timing ratios

Cycle period (CP)

Features are aggregated across a window (e.g., 30 seconds) using:

mean

variance

This produces a rich, physiologically meaningful feature vector per PPG window.

3. Machine Learning Pipeline
Feature Pruning

Features with correlation > 0.95 are removed.

Models Trained

ElasticNet Regression

Gradient Boosting Regressor (GBM)

Random Forest Regressor

LightGBM Regressor

Dummy Mean Predictor (baseline)

Cross-Validation

Uses leave-k-subjects-out:

k ∈ {1, 2, 3}


This mimics realistic generalization to unseen subjects — a key requirement in physiological ML.

4. ML Results (EVAL Dataset)

Metrics computed for each model:

MAE (Mean Absolute Error)

RMSE

MAPE

Standard deviations across folds

Best-performing models for SBP and DBP are included in:

results/YYYY-MM-DD_best_results_eval.csv

5. Clinical Validation (Kolkata ABPM Dataset)

Implemented in KolkataStatsTests.ipynb

This notebook performs clinical-grade validation using ABPM (Ambulatory Blood Pressure Monitoring) reference measurements.

Includes:
Bland–Altman Analysis

For each subject and overall dataset:

Mean difference

Limits of agreement

Outlier count

Visualization saved as PNGs

Dipping Analysis

Evaluates nighttime dipping by computing:

Slope of BP during dipping window

Predicted vs. reference slope comparison

Start/end times of dipping

of points in dipping domain
Error Distribution Analysis

For each subject:

Mean error (bias)

MAE

RMSE

MAPE

% data within 1/2/3 standard deviations

Outlier spread plots

Demographic Correlation

Links error patterns to:

Age

Gender

Medical group

HbA1C

Comorbidities

Medications

Subject-Level Profiles

Identifies:

Best and worst performing subjects

Clinical characteristics of these groups

This section transforms the ML results into clinically interpretable insights, essential for real-world adoption.

6. Data Sources
AIME 2020 EVAL Dataset

Processed 30-second PPG windows (min–max normalized, backfilled).
Citation:
Morassi Sasso, Ariane (2020). Processed EVAL Dataset.
https://doi.org/10.6084/m9.figshare.12649691

Kolkata ABPM Dataset

Multisubject ambulatory BP study used for downstream clinical evaluation.

7. Reproducing the Pipeline
Extract Features
path = extract_features(
    csv=True,
    time_delta='30 seconds',
    time_delta_type='bfill',
    experiment_type='eval',
    special_filter='norm',
    verbose=False
)

Train & Evaluate Models
df = pd.read_csv(path)
results = predict_bp_from_ppg(df, predicted_variable='SBP', k=2)

Run Clinical Validation
# Generate Bland–Altman plots
generateBlandAltmanGraphs(1, 30)

# Compute dipping statistics
generateDippingGraphStats(False)

# Subject-level error stats
generateMeanStats('by subject', False)

8. Key Strengths of This Project

Combines signal processing, machine learning, and clinical statistics

Produces interpretable, physiologically meaningful features

Uses subject-wise validation — far more realistic than random splits

Includes clinical-grade evaluation, uncommon in student ML projects

End-to-end pipeline demonstrates real-world product thinking

9. Validation Report Included

The Validation Report.pdf summarizes:

Best model performance

Error ranges for SBP and DBP

Clinical agreement (LOA)

Interpretation of results

Dipping and circadian behavior analysis

10. Future Work

Possible extensions:

Add waveform decomposition (WBP, second derivative PPG)

Try neural models (TCN, mini-Transformer)

Add motion artifact filtering

Evaluate model personalization approaches

Test on MIMIC-III waveform database

Contributors

Sriram Bhimaraju
Signal Processing · Machine Learning · Clinical Analytics

Feel free to reach out for collaboration!
