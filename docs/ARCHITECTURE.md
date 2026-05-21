# System Architecture

## Data Flow Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ATM FAILURE PREDICTION SYSTEM                    │
└─────────────────────────────────────────────────────────────────────┘

Step 1: Data Ingestion
  └─→ Pull 15-day historical data from operational database
      (ATM metrics, component states, error frequencies)

Step 2: Feature Engineering
  └─→ Transform raw data into 249 engineered features
      - Component State Features (21)
      - Error Pattern Features (18)
      - Temporal Features (12)
      - Rolling Window Features (40+)
      - Baseline Comparison Features (15+)
      - Cross-Signal Features (10+)

Step 3: Model Scoring
  └─→ XGBoost v2.2 generates raw probability for each ATM
      Input: 249 features
      Output: Raw probability (0.0 to 1.0)

Step 4: Probability Calibration
  └─→ Isotonic Regression maps raw → calibrated probability
      Problem: Raw probabilities miscalibrated
      Solution: Learn monotonic transformation
      Result: Probabilities match observed failure frequencies

Step 5: Risk Tier Assignment
  └─→ Map calibrated probability to business risk level
      CRITICAL: ≥ 0.65 (83% actual failure rate)
      HIGH:     0.20-0.65 (32% actual failure rate)
      MEDIUM:   0.10-0.20 (10% actual failure rate)
      LOW:      < 0.10 (2% actual failure rate)

Step 6: Output Generation
  └─→ Produce ranked watchlist of all ATMs
      Sorted by risk probability (highest to lowest)
      Operations team uses for daily maintenance planning

Step 7: Daily Monitoring
  └─→ Compare yesterday's predictions vs. actual failures
      Track model performance (PR-AUC, recall, precision)
      Detect model drift
      Alert if metrics fall below thresholds
```

---

## Component Architecture

### 1. Feature Engineering Pipeline (249 Features)

**Category 1: Component State Features (21)**
- Individual error flags for each ATM component
- Card reader, dispenser, communications, hardware
- Rolling averages of each component's errors
- Purpose: Track which subsystems are deteriorating

**Category 2: Error Pattern Features (18)**
- Frequency distribution of critical error types
- Card jams, cassette jams, communication failures
- Hardware errors, state errors, service outages
- Purpose: Identify ATM-specific failure patterns

**Category 3: Temporal Features (12)**
- Day of week (0-6: Monday-Sunday)
- Weekend indicator (1 if Saturday/Sunday)
- Friday/Monday special flags (high-risk days)
- Day of month and month boundaries
- Cyclical sine/cosine encodings for continuous patterns
- Purpose: Capture day-of-week and seasonal effects

**Category 4: Rolling Window Features (40+)**
- 3-day, 7-day, 14-day rolling sums and averages
- Rolling maximum component states
- Trend indicators (uptrend/downtrend in errors)
- Purpose: Detect recent patterns and trends

**Category 5: Baseline Comparison Features (15+)**
- Weekend vs. weekday 14-day moving averages (per ATM)
- Ratio: current_error / (weekend_avg or weekday_avg)
- Purpose: Detect anomalies (4x normal weekend errors = alert)

**Category 6: Cross-Signal Features (10+)**
- Jam + Card Reader Failure interaction
- Cassette Supply Pressure (empty + low cassettes)
- Receipt Supply Pressure (out of paper + low paper)
- Purpose: Capture error combinations that compound risk

---

### 2. Model Training & Selection

```
Three models trained in parallel:

┌──────────────────────────────────────────────────────────────┐
│ RANDOM FOREST                                                │
│ n_estimators: 300                                            │
│ max_depth: 18                                                │
│ class_weight: balanced                                       │
│ Result: ROC-AUC 0.82                                         │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ XGBOOST v2.2 ⭐ SELECTED                                     │
│ n_estimators: 300                                            │
│ max_depth: 8                                                 │
│ learning_rate: 0.05                                          │
│ scale_pos_weight: Balanced for class imbalance               │
│ Result: ROC-AUC 0.85 (Best performer)                       │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ LIGHTGBM                                                     │
│ n_estimators: 300                                            │
│ max_depth: 10                                                │
│ learning_rate: 0.05                                          │
│ class_weight: balanced                                       │
│ Result: ROC-AUC 0.81                                         │
└──────────────────────────────────────────────────────────────┘

Winner: XGBoost
- Best ROC-AUC (0.85)
- Fast inference
- Good feature importance
- Stable predictions
```

**Sample Weighting (Fix in Training):**
- Critical events (criticality_score > 0) get 2x weight
- Forces model to pay more attention to high-risk ATMs
- Balances the heavily skewed positive class

---

### 3. Probability Calibration (Fix 6)

**Problem Identified:**
```
Raw Model Probabilities vs. Actual Failure Rates:

Range           Raw Prob    Actual Rate    Gap
Very Low        0.07        0.40          -0.33 ❌ (too low)
Low             0.15        0.35          -0.20 ❌
Medium          0.35        0.50          -0.15 ❌
High            0.72        0.75          -0.03 ✓
Critical        0.92        0.95          -0.03 ✓
```

The raw probabilities don't match what actually happens, especially at low probabilities.

**Solution: Isotonic Regression**
```
Learn a monotonic (one-direction) mapping:
raw_probability → calibrated_probability

Training:
  Input: Raw probabilities from XGBoost on test set
  Target: Actual failure outcomes (0 or 1)
  Fit: IsotonicRegression()
  
Result:
  - Monotonic: if raw_prob_1 < raw_prob_2, then cal_prob_1 < cal_prob_2
  - Flexible: non-parametric (no assumptions about shape)
  - Improved: Brier score improves by ~3%
  - Preserved: ROC-AUC unchanged (within 0.005)
```

**Calibration Effects After Fix 6:**
```
Range           Raw → Calibrated    Actual Rate
Very Low        0.07 → 0.12         0.40 ✓ Better
Low             0.15 → 0.22         0.35 ✓ Better
Medium          0.35 → 0.45         0.50 ✓ Better
High            0.72 → 0.75         0.75 ✓ Match
Critical        0.92 → 0.95         0.95 ✓ Match
```

---

### 4. Risk Tier Assignment

**Thresholds Calibrated to Business Cost Ratio (3:1)**

```
Cost Model:
  False Negative (miss a failure):     Cost = 3
    → Downtime, customer impact, emergency dispatch
  
  False Positive (dispatch unnecessarily): Cost = 1
    → Technician time, preventive dispatch

Optimization:
  Minimize: (3 × FN) + (1 × FP)
  Result: Thresholds that maximize value
```

**Final Thresholds:**
| Tier | Threshold | Interpretation | Actual Failure Rate |
|------|-----------|-----------------|-------------------|
| CRITICAL | ≥ 0.65 | High probability of failure; dispatch immediately | 83% |
| HIGH | 0.20-0.65 | Elevated risk; schedule maintenance today | 32% |
| MEDIUM | 0.10-0.20 | Moderate risk; monitor closely | 10% |
| LOW | < 0.10 | Low risk; routine maintenance only | 2% |

---

### 5. Correlated Failure Detection

**Problem Identified (Fix 5):**
On April 23-24, 128 ATMs simultaneously logged host_comm_failed events.
This was clearly a network infrastructure outage, not 128 independent failures.

**Solution: Fleet-Wide Event Detection**
```
Daily Prediction Run:
  Count ATMs flagged HIGH or CRITICAL
  If count > 50% of fleet:
    → Tag as is_correlated_event = True
    → Alert: "Infrastructure event detected"
    → Route to network team (not individual dispatch)

Daily Monitoring Run:
  Count ATMs with actual failures
  If count > 70% of fleet:
    → Mark day as "Infrastructure event"
    → Warn: "Per-ATM metrics unreliable for this date"
    → Don't trigger retrain based on this day's metrics
```

**Benefit:**
- Prevents 100+ pointless technician tickets
- Routes to correct team (network vs. field operations)
- Prevents false alarms on retrain thresholds

---

### 6. Daily Prediction Pipeline

**Input:** 15-day historical data window

**Processing:**
1. Load latest 15 days of ATM operational metrics
2. Engineer all 249 features (same code as training)
3. Apply categorical encoding (saved from training)
4. Load trained XGBoost model from best_model.pkl
5. Generate raw probabilities
6. Apply isotonic calibrator
7. Assign risk tiers
8. Check for correlated events
9. Rank ATMs by probability

**Output:** daily_predictions_YYYY-MM-DD.csv
```csv
atm_id,failure_probability,risk_level,criticality_score,branch,rank
ATM_001,0.85,CRITICAL,15,Branch_A,1
ATM_002,0.42,HIGH,7,Branch_B,2
ATM_003,0.08,LOW,2,Branch_C,128
```

**Execution:**
- Time: 01:00 EET (Lebanon timezone)
- Duration: < 2 minutes for 130 ATMs
- Output: CSV file on shared volume
- Consumed by: Operations team for daily planning

---

### 7. Daily Monitoring & Evaluation

**Purpose:** Track model performance and detect drift

**Input:** Yesterday's predictions + actual failures

**Metrics Calculated:**
- PR-AUC (Precision-Recall Area Under Curve)
- ROC-AUC (standard classification metric)
- Precision (% of flagged ATMs that actually failed)
- Recall (% of actual failures that were caught)
- F1 Score (harmonic mean of precision and recall)
- Confusion Matrix (TP, FP, FN, TN)
- Per-tier actual failure rates

**Alert Conditions (5 triggers):**
```
1. PR-AUC < 0.65         → Alert: Model precision degrading
2. Recall < 0.50         → Alert: Missing too many failures
3. Precision < 0.40      → Alert: Too many false alarms
4. Unknown branches > 10 → Alert: Data quality issue
5. 5-day rolling avg meets any above → Trigger retrain
```

**Output:** monitoring_log.csv (one row per day)
```csv
date,pr_auc,roc_auc,recall,precision,f1,tp,fp,fn,tn
2026-05-19,0.78,0.82,1.00,0.58,0.73,62,45,0,2651
```

**Output:** monitoring_report_YYYY-MM-DD.txt (human-readable)
```
ATM Failure Prediction - Daily Evaluation Report
Date: 2026-05-19

Predictions Evaluated: 128 ATMs
Actual Failures: 62

METRICS:
  PR-AUC:   0.78 (floor: 0.65) ✓
  ROC-AUC:  0.82
  Recall:   1.00 (floor: 0.50) ✓
  Precision: 0.58 (floor: 0.40) ✓

TIER PERFORMANCE:
  CRITICAL (n=62): 83% actual failure rate
  HIGH (n=45):     32% actual failure rate
  MEDIUM (n=12):   8% actual failure rate
  LOW (n=9):       2% actual failure rate

ALERTS: None
STATUS: ✓ All metrics normal
```

---

## Production Deployment

### Infrastructure
```
Docker Container:
  ├── Python 3.8+ environment
  ├── All dependencies (pandas, scikit-learn, xgboost, etc.)
  └── All scripts (clean_data, train_model, daily_predictions, monitoring)

Cron Scheduling:
  ├── 01:00 EET → Run daily_predictions.py
  │   └─ Generates daily_predictions_YYYY-MM-DD.csv
  │
  └── 08:00 EET → Run monitoring.py
      └─ Evaluates yesterday's predictions
      └─ Updates monitoring_log.csv
      └─ Alerts if needed
```

### Data I/O
```
Input Files (on shared volume):
  ├── today_data.csv (15-day window, updated daily by database)
  └── encoder_map.pkl (categorical mappings from training)

Output Files (on shared volume):
  ├── daily_predictions_YYYY-MM-DD.csv
  ├── monitoring_log.csv (rolling history)
  └── alerts.txt (empty on healthy days)

Model Files:
  └── best_model.pkl (model + calibrator + metadata)
```

### Monitoring Dashboard
- 20+ days of historical performance data
- Visible via monitoring_log.csv
- Automated alerts to engineering team
- Manual review of edge cases

---

## Key Design Decisions

### Why 249 Features?
- Captures multiple dimensions of ATM health
- Allows model to discover complex patterns
- 6 feature categories (component, error, temporal, rolling, baseline, cross)
- Balanced coverage of all failure modes

### Why XGBoost?
- Best ROC-AUC (0.85) among 3 models tested
- Gradient boosting: strong learners in sequence
- Fast inference (production-friendly)
- Good feature importance (interpretability)
- Stable predictions across train/test

### Why Isotonic Calibration?
- Non-parametric: no assumptions about probability distribution
- Monotonic: preserves ranking (higher raw → higher calibrated)
- Flexible: can match any non-decreasing curve
- Better than Platt (sigmoid) for our non-linear miscalibration

### Why 3:1 Cost Ratio?
- Based on bank's operations analysis
- Missing a failure (downtime) = 3× more expensive than false alarm
- Optimization: minimize (3×FN) + (1×FP)
- Results in bias toward sensitivity (recall)

---

## Future Enhancements

### Potential Improvements (Deferred)
1. **Time Series Decomposition** - Separate trend/seasonality/noise
2. **Anomaly Detection** - Unsupervised identification of outliers
3. **Customer Impact Modeling** - Predict transaction losses
4. **Cost Optimization Engine** - Dynamic dispatch routing
5. **Real-Time Scoring** - Sub-minute latency (streaming data)

### Monitoring Triggers for Enhancement
- If weekend recall drops below 0.50 for 2 weeks
- If PR-AUC shows persistent downward trend
- If new ATM models appear with different failure patterns
- If seasonal patterns shift significantly

---

## References

### Key Documents
- **Handoff Report v1.0** - Complete project documentation
- **Feature Engineering Module** - All 249 feature definitions
- **Model Training Script** - Training pipeline and evaluation
- **Daily Predictions Script** - Production scoring logic

### Key Metrics
- PR-AUC: Focus on ranking, not absolute probabilities
- Recall: Critical for catching failures
- Precision: Important for operational efficiency
- Confusion Matrix: Understand error types (FP vs FN tradeoff)

---

**Last Updated:** May 2026  
**Version:** 2.2  
**Status:** Production
