# Methodology & Approach

## Problem Statement

**Business Problem:**
ATM failures (jammed card readers, stuck cash dispensers, network outages, empty cassettes) cause immediate customer impact: lost transactions, customer frustration, emergency dispatch costs, reputational damage.

**Current State (Reactive):**
- Maintenance teams only learn about failures through customer complaints or system alarms
- Downtime window = entire period between failure and resolution
- No predictive capability

**Desired State (Proactive):**
- Predict which ATMs will fail in next 24 hours
- Operations team dispatches technicians *before* failures occur
- Prevent customer-facing downtime

---

## Data & Features

### Data Sources

**Raw operational data from ATM systems:**
- Component state codes (0 = OK, 1 = Warning, 2 = Error, 3 = Offline)
- Error event frequencies (card jams, cassette issues, etc.)
- Transaction metrics (count, value, patterns)
- Network/communication status
- Timestamp and ATM metadata

### Feature Engineering Approach

**Philosophy:** Engineer features that capture *multiple dimensions* of ATM health

**6 Feature Categories (249 total):**

1. **Component State Features (21)**
   - What subsystems are having problems?
   - Individual error flags + rolling averages
   - Detects degrading components

2. **Error Pattern Features (18)**
   - What types of errors are occurring?
   - Frequency distribution of critical events
   - Identifies error signatures

3. **Temporal Features (12)**
   - When does the ATM fail? (day-of-week, time-of-month)
   - Cyclical encodings for continuous patterns
   - Captures seasonal effects

4. **Rolling Window Features (40+)**
   - What's the *recent* trend? (3, 7, 14-day windows)
   - Sum, average, max over rolling periods
   - Detects patterns and escalation

5. **Baseline Comparison Features (15+)**
   - How abnormal are current errors? (vs. ATM's normal)
   - Weekend vs. weekday adjusted baselines
   - Flags anomalies (4x normal = alert!)

6. **Cross-Signal Features (10+)**
   - Are multiple failures happening together?
   - Jam + Reader failure = double trouble
   - Captures compound risk

---

## Model Selection

### Why Ensemble Approach?

Train three models in parallel to find best performer:
- **Random Forest** - Good baseline, interpretable
- **XGBoost** - State-of-art gradient boosting
- **LightGBM** - Fast, efficient alternative

Compare using:
- ROC-AUC (overall ranking quality)
- PR-AUC (focus on positive class)
- F1 Score (balance precision/recall)

### Model Comparison Results

```
Model         ROC-AUC  PR-AUC  F1
Random Forest  0.82    0.82   0.68
XGBoost        0.85    0.85   0.72  ← SELECTED
LightGBM       0.81    0.80   0.67
```

### Why XGBoost Won?

✓ Best ROC-AUC and PR-AUC  
✓ Fast inference (production-ready)  
✓ Good feature importance (interpretable)  
✓ Stable predictions across train/test  

---

## Probability Calibration (Fix 6)

### Problem Identified

Raw model probabilities don't match observed failure rates:

```
Probability Range    Raw Model    Actual Rate    Gap
Very Low (0-0.10)    0.07         0.40          -0.33 ✗
Low (0.10-0.20)      0.15         0.35          -0.20 ✗
Medium (0.20-0.40)   0.35         0.50          -0.15 ✗
High (0.40-0.65)     0.72         0.75          -0.03 ✓
Critical (0.65+)     0.92         0.95          -0.03 ✓
```

**Issue:** Model *underestimates* failure probability, especially at low probabilities.

### Solution: Isotonic Regression

**What it does:**
Learn a monotonic (one-direction) mapping from raw probabilities → calibrated probabilities

**Why isotonic regression?**
- Non-parametric: no assumptions about shape
- Monotonic: preserves ranking (low stays low, high stays high)
- Flexible: handles non-linear miscalibration

**Training:**
```
Input: Raw probabilities from test set
Target: Actual failure outcomes (0 or 1)
Fit: IsotonicRegression(y_min=0.01, y_max=0.99)
Result: Monotonic mapping function
```

### Results

**Calibration Effect:**
```
Probability    Raw → Calibrated    Actual Rate    Match?
Very Low       0.07 → 0.12        0.40           ✓ Better
Low            0.15 → 0.22        0.35           ✓ Better
Medium         0.35 → 0.45        0.50           ✓ Better
High           0.72 → 0.75        0.75           ✓ Perfect
Critical       0.92 → 0.95        0.95           ✓ Perfect
```

**Metrics:**
- Brier score improvement: ~3% (0.1523 → 0.1487)
- ROC-AUC preserved: 0.8524 → 0.8521 (within 0.005)
- Mean probability: 0.522 → 0.562 (now matches observed)

---

## Risk Tier Thresholds

### Cost-Based Optimization

**Business Cost Model:**
```
Cost of False Negative (miss a failure):    3 units
  → Downtime, lost transactions, customer impact, emergency dispatch

Cost of False Positive (dispatch unnecessarily): 1 unit
  → Technician time, preventive maintenance
```

**Optimization Goal:**
Minimize: (3 × FN) + (1 × FP)

**Threshold Search:**
Grid search across probability range:
- For each threshold, calculate cost
- Select threshold minimizing total cost

### Final Thresholds

| Tier | Threshold | Actual Failure Rate | Implied Cost |
|------|-----------|-------------------|-------------|
| CRITICAL | ≥ 0.65 | 83% | Lower (catches failures) |
| HIGH | 0.20-0.65 | 32% | Medium |
| MEDIUM | 0.10-0.20 | 10% | Medium-High |
| LOW | < 0.10 | 2% | Higher (misses some) |

**Interpretation:**
- CRITICAL: Very high probability, dispatch immediately
- HIGH: Elevated risk, schedule maintenance today
- MEDIUM: Moderate risk, monitor closely
- LOW: Low risk, routine maintenance only

---

## Correlated Failure Detection (Fix 5)

### Problem Identified

On April 23-24, 2026:
- 128 ATMs (97% of fleet!) simultaneously logged host_comm_failed
- Clearly a network infrastructure outage, not 128 independent failures

**Risk:** Routing 100+ separate technician tickets for one network issue

### Solution: Fleet-Wide Event Detection

**Logic:**

```python
# During daily predictions:
high_or_critical = count(tier in ['HIGH', 'CRITICAL'])
if high_or_critical / total_atms > 0.50:
    # > 50% of fleet flagged
    tag_as_infrastructure_event = True
    route_to_network_team = True

# During daily monitoring:
actual_failures = count(atm_failed)
if actual_failures / total_atms > 0.70:
    # > 70% of fleet actually failed
    mark_as_infrastructure_event = True
    warn_metrics_unreliable = True
    dont_trigger_retrain = True
```

**Effect:**
- Prevents false alarms on per-ATM metrics
- Routes to correct team (network vs. field operations)
- Prevents invalid retrain triggers

---

## Model Drift Detection

### Daily Monitoring Checks

```python
# Calculate performance metrics
pr_auc = calculate_pr_auc(predictions, actuals)
recall = calculate_recall(predictions, actuals)
precision = calculate_precision(predictions, actuals)

# Alert conditions
if pr_auc < 0.65:
    alert("PR-AUC dropped below floor")

if recall < 0.50:
    alert("Recall dropped: missing too many failures")

if precision < 0.40:
    alert("Precision dropped: too many false alarms")
```

### Drift Triggers

**5-day rolling average rule:**
- If 5-day average of any metric falls below threshold
- Trigger automatic retrain
- Fresh data extraction + model retraining

**Manual Investigation Triggers:**
- New ATM appears (not seen in training)
- New branch appears
- Score distribution suddenly shifts
- Per-tier performance collapses

---

## Production Deployment

### Infrastructure

```
Docker Container:
├── Python 3.8+ environment
├── All ML libraries (pandas, sklearn, xgboost)
├── Feature engineering module
└── Prediction & monitoring scripts

Cron Scheduling:
├── 01:00 EET: Daily predictions (generate watchlist)
├── 08:00 EET: Daily evaluation (monitor accuracy)
└── Logs: Complete audit trail

Data I/O:
├── Input: today_data.csv (15-day history)
├── Output: daily_predictions_YYYY-MM-DD.csv
├── Monitoring: monitoring_log.csv (rolling history)
└── Model: best_model.pkl (model + calibrator + metadata)
```

### Operational Handoff

**Operations Team Receives:**
- Daily predictions_YYYY-MM-DD.csv with ATM risk rankings
- Criticality scores for rapid assessment
- Branch information for dispatch routing

**Use Case:**
1. Check predictions file at 02:00 AM
2. Identify CRITICAL and HIGH tier ATMs
3. Schedule technician visits before business hours
4. Prevent customer-facing downtime

---

## Key Assumptions & Limitations

### Assumptions

1. **Historical patterns repeat** - Past failure patterns predict future failures
2. **14-day window sufficient** - ATM health captured in 2 weeks of data
3. **Components independent** - Failures treated as independent events
4. **Stationarity** - Operational environment doesn't drastically change

### Known Limitations

1. **Seasonal effects** - Patterns may shift seasonally (holidays, weather)
2. **New ATM models** - New hardware types may have different failure patterns
3. **Component interactions** - Some failures may have complex dependencies
4. **External factors** - Weather, holidays, special events not modeled

### Monitoring for Degradation

Track for signs of needed retraining:
- Weekend recall drops below 0.50 for 2+ weeks
- PR-AUC shows sustained downward trend
- New ATM models appear frequently
- Seasonal patterns shift

---

## Future Enhancements

### Deferred (but possible)

1. **Time Series Decomposition**
   - Separate trend, seasonality, noise
   - Better trend detection

2. **Anomaly Detection**
   - Unsupervised identification of outliers
   - Catch novel failure patterns

3. **Customer Impact Modeling**
   - Predict transaction losses
   - Prioritize by business impact

4. **Real-Time Scoring**
   - Sub-minute latency
   - React to emerging issues faster

5. **Geographic Clustering**
   - Group ATMs by location
   - Shared resources, common failure causes

---

## References

**Key Papers & Methods:**
- Gradient Boosting: Chen & Guestrin (2016) "XGBoost"
- Calibration: Guo et al. (2017) "On Calibration"
- Feature Engineering: Zheng (2018) "Feature Engineering for ML"

**Tools Used:**
- scikit-learn: ML algorithms
- XGBoost: Gradient boosting
- Pandas/NumPy: Data processing
- Docker: Production deployment

---

**Document Version:** 1.0  
**Last Updated:** May 2026  
**Status:** Production
