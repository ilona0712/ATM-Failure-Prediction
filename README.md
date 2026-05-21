# ATM Failure Prediction System

**Production ML pipeline for predictive maintenance of ATM networks**

An end-to-end machine learning system that predicts ATM failures 24 hours in advance using advanced feature engineering, ensemble models, and real-time monitoring.

## 🎯 Project Overview

This system transformed ATM maintenance from **reactive** (fixing failures after they occur) to **proactive** (preventing failures before customers are affected).

### Key Achievement
- **v2.2 Production Model**: XGBoost ensemble with 249 engineered features
- **PR-AUC**: 0.85 on validation set
- **Recall**: 81% average (weekday), 71% (weekend)
- **Daily Deployment**: Automated scoring of 130+ ATMs
- **Infrastructure Event Detection**: Identifies fleet-wide outages vs. individual failures

## 🏗️ System Architecture

```
┌─────────────────┐
│  Raw Data       │
│  (SQL Server)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  Feature Engineering (249)  │
│  • Component states         │
│  • Error patterns           │
│  • Temporal features        │
│  • Rolling windows (3/7/14d)│
│  • Cross-signal analysis    │
└────────┬────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Model Training              │
│  • Random Forest             │
│  • XGBoost v2.2 (selected)  │
│  • LightGBM                  │
│  • Isotonic Calibration      │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Daily Predictions           │
│  • Risk scoring              │
│  • Tier assignment           │
│  • Correlated event detection│
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Monitoring & Evaluation     │
│  • Prediction accuracy       │
│  • Model drift detection     │
│  • Alert generation          │
└──────────────────────────────┘
```

## 📊 Feature Engineering (249 Features)

### Component State Features (21)
Flags and rolling averages for each ATM component:
- Card reader errors
- Dispenser jams and failures
- Communication errors
- Hardware failures

### Error Pattern Features (18)
Frequency distribution of critical events:
- ATM-specific error signatures
- Error severity scoring
- Component interaction effects

### Temporal Features (12)
Day-of-week and cyclical patterns:
- Day of week (0-6)
- Weekend flag
- Friday/Monday indicators
- Cyclical sine/cosine encoding

### Rolling Window Features (40+)
7-day, 14-day, and 30-day statistics:
- Error count sums and averages
- Max component states
- Trend indicators

### Baseline Comparison Features (15+)
Weekend vs. weekday relative metrics:
- Component error ratios
- Adjusted for ATM-specific baselines
- Detects anomalies in normal patterns

### Cross-Signal Features (10+)
Interactions between error types:
- Jam + reader failure correlation
- Cassette supply pressure
- Receipt supply pressure

## 🤖 Models & Selection

Three ensemble models trained in parallel:

| Model | Type | PR-AUC | Notes |
|-------|------|--------|-------|
| Random Forest | Bagging | 0.82 | Good interpretability |
| **XGBoost** | **Boosting** | **0.85** | **Selected - Best overall** |
| LightGBM | Gradient Boost | 0.81 | Fast inference |

### Winner: XGBoost v2.2
- 300 trees, max_depth=8, learning_rate=0.05
- Scale positive weights for class imbalance
- Sample weights boost critical-event rows by 2x

## 🎚️ Probability Calibration

**Fix #6: Isotonic Regression**

The raw model probabilities were miscalibrated:
- Weekend probabilities compressed (too low)
- Critical probabilities stretched (too high)

Solution: Isotonic regression learns a monotonic mapping from raw→calibrated probability.

**Results:**
- Brier score improved by ~3%
- ROC-AUC preserved (within 0.005)
- Probabilities now match observed frequencies

## 📈 Risk Tiers

Probabilities mapped to business-calibrated risk levels:

| Tier | Threshold | Interpretation |
|------|-----------|-----------------|
| **CRITICAL** | ≥ 0.65 | High probability of failure; dispatch immediately |
| **HIGH** | 0.20 - 0.65 | Elevated risk; schedule maintenance |
| **MEDIUM** | 0.10 - 0.20 | Moderate risk; monitor closely |
| **LOW** | < 0.10 | Low risk; routine maintenance only |

Thresholds calibrated to 3:1 cost ratio (missing a failure is 3× more expensive than unnecessary dispatch).

## 🚨 Correlated Failure Detection

When >50% of fleet flagged HIGH/CRITICAL on same day:
- Tag as potential infrastructure event
- Alert network team (not individual technicians)
- Different escalation path

Example: Network outage on 2026-04-23 affected 128 ATMs simultaneously.

## 📊 Daily Predictions

### Input
- 15-day historical data for rolling window computation
- Today's ATM operational metrics

### Output
**`daily_predictions_YYYY-MM-DD.csv`** with:
- `atm_id`: ATM identifier
- `failure_probability`: Calibrated probability (0-1)
- `risk_level`: CRITICAL / HIGH / MEDIUM / LOW
- `criticality_score`: Component state summary
- `critical_incident_score`: Active error severity
- `branch`: ATM branch location
- `rank`: Risk ranking within fleet

Example:
```
atm_id,failure_probability,risk_level,criticality_score,critical_incident_score,branch
ATM_DEMO_001,0.85,CRITICAL,15,8,Branch_A
ATM_DEMO_002,0.42,HIGH,7,2,Branch_B
ATM_DEMO_003,0.08,LOW,2,0,Branch_C
```

## 📡 Monitoring System

### Daily Evaluation
Compares yesterday's predictions to actual failures:
- PR-AUC tracking
- Recall and precision
- Confusion matrix
- Score distribution

### Alert Conditions
Auto-triggers retraining if:
1. PR-AUC drops below 0.65
2. Recall drops below 0.50
3. Precision drops below 0.40
4. Unknown branches exceed 10
5. 5-day rolling average meets any condition

### Sample Monitoring Report
```
Date: 2026-05-19
Predictions: 128 ATMs evaluated
PR-AUC: 0.78
Recall: 1.00 (all failures caught)
Precision: 0.58 (58% of flagged ATMs actually failed)

CRITICAL tier: 83% actual failure rate
HIGH tier: 32% actual failure rate
```

## 🔧 Getting Started

### Requirements
```bash
pip install -r requirements.txt
```

Python 3.8+, with dependencies:
- pandas, numpy (data manipulation)
- scikit-learn, xgboost, lightgbm (ML)
- joblib (model serialization)
- shap (interpretability)
- pytz (timezone support)

### File Structure
```
ATM-Failure-Prediction/
├── README.md                          (this file)
├── requirements.txt                   (Python dependencies)
├── data/
│   ├── sample_data_input.csv         (15-day window example)
│   └── sample_predictions_output.csv (example predictions)
├── src/
│   ├── feature_engineering.py        (shared feature logic)
│   ├── model_training.py             (train.py equivalent)
│   ├── daily_predictions.py          (score.py equivalent)
│   └── monitoring.py                 (eval.py equivalent)
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
└── docs/
    ├── ARCHITECTURE.md
    ├── METHODOLOGY.md
    └── TROUBLESHOOTING.md
```

### Quick Demo

1. **Load sample data and generate predictions:**
```bash
python src/daily_predictions.py --input data/sample_data_input.csv
```

2. **Retrain model on sample data:**
```bash
python src/model_training.py --train-file data/sample_training.csv
```

3. **Run monitoring evaluation:**
```bash
python src/monitoring.py --predictions-file data/sample_predictions.csv
```

4. **Explore the analysis:**
```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

## 📚 Key Scripts

### `feature_engineering.py`
Shared feature engineering module used by both training and scoring:
- Critical event scoring (weighted by severity)
- Temporal features (day-of-week, cyclical)
- Rolling window computations
- Baseline comparison ratios
- Cross-signal interactions

### `model_training.py`
Trains ensemble of models and selects best:
- Loads train/test data from clean_data.py output
- Trains Random Forest, XGBoost, LightGBM in parallel
- Evaluates with PR-AUC, ROC-AUC, F1, confusion matrix
- Applies sample weights for critical events
- Fits isotonic calibration on test set
- Bundles complete model package with features + encoders

### `daily_predictions.py`
Scores all ATMs daily with reproducible features:
- Loads latest 15 days of historical data
- Computes rolling windows (same logic as training)
- Applies categorical encoding (saved from training)
- Generates raw probabilities from XGBoost
- Applies isotonic calibration
- Maps to risk tiers
- Detects correlated failure events
- Outputs ranked watchlist

### `monitoring.py`
Daily evaluation of model performance:
- Compares predictions to actual failures
- Calculates PR-AUC, ROC-AUC, precision, recall
- Tracks confusion matrix
- Detects model drift
- Generates alerts if thresholds breached
- Logs performance history

## 🔍 Model Interpretability

### Feature Importance
Top features for XGBoost v2.2:

1. `day_of_week` - Temporal pattern strong predictor
2. `dispenser_not_operational_7d` - Recent dispenser issues
3. `is_friday` - Friday-specific risk
4. `cdm_error_vs_dow_baseline` - Dispenser error ratio
5. `card_reader_jammed_7d` - Reader jam frequency

### SHAP Values
Individual predictions explained via SHAP:
```python
import shap
import joblib

model_pkg = joblib.load('models/best_model.pkl')
explainer = shap.TreeExplainer(model_pkg['model'])
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

Shows which features pushed each ATM's score up/down.

## 📊 Results & Impact

### Production Metrics
- **Daily execution**: 01:00 EET predictions, 08:00 EET evaluation
- **Latency**: <2 minutes for 130 ATMs
- **Uptime**: 99.8% (20+ days consecutive operation)
- **Model drift**: None detected (5-day rolling average stable)

### Business Impact
- Maintenance proactive instead of reactive
- Daily watchlist prevents unplanned downtime
- Infrastructure events identified early
- Cost-calibrated risk tiers guide dispatch priorities
- SHAP interpretability builds trust with operations team

## 🔐 Data Privacy

This repository contains:
- ✅ Feature engineering approach
- ✅ Model training methodology
- ✅ Calibration techniques
- ✅ Monitoring system design
- ❌ NO real bank data
- ❌ NO database credentials
- ❌ NO real ATM IDs or branch locations
- ❌ NO real performance metrics (if confidential)

All data in this repo is synthetic/example data for demonstration.

## 📖 Documentation

- **ARCHITECTURE.md** - System design and data flow
- **METHODOLOGY.md** - Feature engineering and modeling approach
- **TROUBLESHOOTING.md** - Common issues and solutions

## 🚀 Deployment Notes

### Production Environment
- Docker container (reproducible environment)
- Cron scheduling (01:00, 08:00 EET)
- File-based input/output (CSV on shared volume)
- Automated via Airflow DAG

### Monitoring & Alerting
- 5-day rolling average for metric smoothing
- PR-AUC floor: 0.65
- Recall floor: 0.50
- Unknown branch ceiling: 10
- Auto-retrain trigger if 5-day rolling fails any condition

## 🔄 Continuous Improvement

### Next Steps
- [ ] Two-week observation window for weekend recall
- [ ] Quarterly retrain (fresh data extraction)
- [ ] Enhance feature engineering with domain feedback
- [ ] Integration with operations scheduling system

### Known Limitations
- Features strongly depend on temporal patterns (may shift seasonally)
- Correlated failure threshold (50%) may need adjustment by domain expert
- Real performance varies by ATM population and branch
- Requires continuous monitoring in production

## 📞 Questions?

This is a sanitized, educational version of a production system. For implementation in your environment, ensure you:
- Adapt feature engineering to your data schema
- Validate thresholds with your domain experts
- Establish proper monitoring and alerting
- Plan for regular retraining with fresh data

## 📜 License

MIT License - feel free to use for learning and reference.

---

**Note**: This project demonstrates:
- Advanced feature engineering (249 features from raw operational data)
- Ensemble model selection and comparison
- Probability calibration for business alignment
- Production deployment and monitoring
- Real-time inference at scale

It is a working example of ML ops best practices applied to predictive maintenance.
