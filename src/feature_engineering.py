"""
ATM Failure Prediction - Feature Engineering Module
====================================================
Shared feature engineering logic used by both training and daily scoring.
This module encapsulates all feature transformations so they're consistent
between model development and production deployment.

Key Design:
  - All feature creation logic in one place
  - Used by both clean_data.py (training) and daily_predictions.py (scoring)
  - Ensures features are identical in training vs. production
  - Handles missing values, infinite values, and encoding

Feature Categories (249 total):
  1. Component State Features (21)
  2. Error Pattern Features (18)
  3. Temporal Features (12)
  4. Rolling Window Features (40+)
  5. Baseline Comparison Features (15+)
  6. Cross-Signal Features (10+)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Critical events with domain-based severity weights
# Weight 3 = service-down (immediate impact)
# Weight 2 = degraded (significant impact)
# Weight 1 = early warning (monitor)
CRITICAL_EVENTS = {
    'card_reader_jammed':               3,   # Cannot read cards
    'cdm_error':                        3,   # Cash dispenser mechanical failure
    'dispenser_not_operational':        3,   # No cash dispensing
    'hardware_error':                   3,   # Critical hardware failure
    'go_out_of_service':                3,   # ATM taken offline
    'cashin_jam':                       2,   # Deposit mechanism stuck
    'cashin_not_operational':           2,   # Cannot accept deposits
    'communication_error':              2,   # Network issues
    'comm_offline':                     2,   # Complete offline
    'chip_contact_error':               2,   # EMV chip reader issues
    'reject_cassette_full':             2,   # Reject cassette capacity
    'total_cassettes_empty':            2,   # No cash remaining
    'fraud_attempts':                   2,   # Security alert
    'jrn_rec_error':                    1,   # Journal/receipt issues
    'receipt_printer_not_operational':  1,   # Cannot print receipts
    'receipt_paper_out':                1,   # Low receipt paper
}

# Rolling window sizes (days)
ROLLING_WINDOWS = [3, 7, 14]

# Columns to compute rolling sums/averages
ROLLING_ERROR_COLS = [
    'total_errors', 'hardware_error', 'cdm_error', 'communication_error',
    'chip_contact_error', 'go_out_of_service', 'card_reader_jammed',
    'dispenser_not_operational', 'comm_offline', 'total_cassettes_empty',
    'critical_incident_score', 'criticality_score',
]

# Columns to compute rolling max
ROLLING_STATE_COLS = ['worst_component_state', 'component_risk_score']


def compute_criticality_score(df):
    """
    Compute weighted criticality score based on active errors.
    
    Logic:
      - For each critical event column, multiply count × severity weight
      - Sum all weighted event counts
      - Result: 0 (no issues) to 100+ (many critical events)
    
    Example:
      1 card_reader_jammed + 2 cdm_errors + 1 communication_error
      = (1×3) + (2×3) + (1×2) = 11
    """
    criticality = pd.Series(0, index=df.index)
    
    for event_col, severity in CRITICAL_EVENTS.items():
        if event_col in df.columns:
            criticality += df[event_col].fillna(0) * severity
    
    return criticality.round(2)


def compute_critical_incident_score(df):
    """
    Alternative criticality metric focusing on immediate incidents.
    Weights incidents by their recency and severity.
    """
    incident_score = pd.Series(0, index=df.index)
    
    # High-severity events (weight 1.0)
    high_severity = [
        'card_reader_jammed', 'cdm_error', 'dispenser_not_operational',
        'hardware_error', 'go_out_of_service'
    ]
    for col in high_severity:
        if col in df.columns:
            incident_score += df[col].fillna(0) * 1.0
    
    # Medium-severity events (weight 0.5)
    medium_severity = [
        'cashin_jam', 'cashin_not_operational', 'communication_error',
        'comm_offline', 'chip_contact_error', 'total_cassettes_empty'
    ]
    for col in medium_severity:
        if col in df.columns:
            incident_score += df[col].fillna(0) * 0.5
    
    return incident_score.round(2)


def create_temporal_features(df):
    """
    Create temporal features capturing day-of-week and seasonal patterns.
    
    Features:
      - day_of_week: 0-6 (Monday-Sunday)
      - is_weekend: 1 if Saturday/Sunday
      - is_friday: Special flag for Friday (high-risk day)
      - is_monday: Special flag for Monday
      - day_of_month: 1-31
      - is_month_start: 1 if first 3 days of month
      - is_month_end: 1 if last 5 days of month
      - dow_sin, dow_cos: Cyclical encoding of day-of-week
      - dom_sin, dom_cos: Cyclical encoding of day-of-month
    
    Cyclical encoding preserves similarity (Friday ≈ Saturday in distance).
    """
    if 'event_date' not in df.columns:
        return df
    
    dt = pd.to_datetime(df['event_date'], errors='coerce')
    
    df['day_of_week'] = dt.dt.dayofweek          # 0=Mon, 4=Fri, 5=Sat, 6=Sun
    df['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
    df['is_friday'] = (dt.dt.dayofweek == 4).astype(int)
    df['is_monday'] = (dt.dt.dayofweek == 0).astype(int)
    df['day_of_month'] = dt.dt.day
    df['is_month_start'] = (dt.dt.day <= 3).astype(int)
    df['is_month_end'] = (dt.dt.day >= 27).astype(int)
    df['is_mid_month'] = ((dt.dt.day >= 14) & (dt.dt.day <= 16)).astype(int)
    
    # Cyclical encoding: convert to angle (0-2π), then sin/cos
    # This makes Friday and Monday closer than Friday and Tuesday
    df['dow_sin'] = np.sin(2 * np.pi * dt.dt.dayofweek / 7).round(6)
    df['dow_cos'] = np.cos(2 * np.pi * dt.dt.dayofweek / 7).round(6)
    df['dom_sin'] = np.sin(2 * np.pi * dt.dt.day / 30).round(6)
    df['dom_cos'] = np.cos(2 * np.pi * dt.dt.day / 30).round(6)
    
    return df


def create_weekend_relative_features(df):
    """
    Create baseline comparison features that adjust for day-of-week patterns.
    
    Logic:
      - For each error type, compute weekend and weekday 14-day moving averages per ATM
      - Create ratio: current_error / (weekend_avg or weekday_avg)
      - Ratios detect anomalies in normal patterns
    
    Example:
      - Normal Saturday: 2 errors, weekend_avg=2.0 → ratio=1.0 (normal)
      - Anomaly Saturday: 8 errors, weekend_avg=2.0 → ratio=4.0 (4x normal!)
    """
    if 'event_date' not in df.columns or 'atm_id' not in df.columns:
        return df
    
    if 'is_weekend' not in df.columns:
        df = create_temporal_features(df)
    
    df = df.sort_values(['atm_id', 'event_date']).reset_index(drop=True)
    
    error_cols = [c for c in df.columns if c in [
        'total_errors', 'hardware_error', 'cdm_error',
        'communication_error', 'card_reader_jammed',
        'dispenser_not_operational', 'comm_offline',
    ]]
    
    grouped = df.groupby('atm_id')
    
    for col in error_cols:
        if col not in df.columns:
            continue
        
        # Weekday 14-day average (only include days where is_weekend=0)
        wd_name = f'{col}_weekday_avg_14d'
        df[wd_name] = grouped[col].transform(
            lambda s: s.where(df.loc[s.index, 'is_weekend'] == 0)
                      .shift(1).rolling(14, min_periods=1).mean()
        ).fillna(0)
        
        # Weekend 14-day average (only include days where is_weekend=1)
        we_name = f'{col}_weekend_avg_14d'
        df[we_name] = grouped[col].transform(
            lambda s: s.where(df.loc[s.index, 'is_weekend'] == 1)
                      .shift(1).rolling(14, min_periods=1).mean()
        ).fillna(0)
        
        # Ratio: current / appropriate baseline
        ratio_name = f'{col}_vs_dow_baseline'
        baseline = np.where(df['is_weekend'] == 1, df[we_name], df[wd_name])
        baseline = np.where(baseline == 0, 1, baseline)  # Avoid division by zero
        df[ratio_name] = (df[col] / baseline).round(4)
    
    return df


def create_rolling_features(df):
    """
    Create rolling window statistics (3-day, 7-day, 14-day).
    
    For each error type, compute:
      - Sum of occurrences over window
      - Average occurrences over window
      - Max component state over window
    
    These capture trends and patterns in recent error history.
    """
    if 'event_date' not in df.columns or 'atm_id' not in df.columns:
        return df
    
    df = df.sort_values(['atm_id', 'event_date']).reset_index(drop=True)
    grouped = df.groupby('atm_id')
    
    # Rolling sums and averages for error columns
    for window in ROLLING_WINDOWS:
        for col in ROLLING_ERROR_COLS:
            if col not in df.columns:
                continue
            
            sum_name = f'{col}_sum_{window}d'
            avg_name = f'{col}_avg_{window}d'
            
            df[sum_name] = grouped[col].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).sum()
            ).fillna(0)
            
            df[avg_name] = grouped[col].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            ).fillna(0)
    
    # Rolling max for state columns
    for window in ROLLING_WINDOWS:
        for col in ROLLING_STATE_COLS:
            if col not in df.columns:
                continue
            
            max_name = f'{col}_max_{window}d'
            df[max_name] = grouped[col].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).max()
            ).fillna(0)
    
    return df


def create_interaction_features(df):
    """
    Create cross-signal features that capture interactions between errors.
    
    Logic:
      - Jam + Card Reader Failure: Can't read cards AND can't dispense
      - Cassette Supply Pressure: Empty cassettes + Low cassettes
      - Receipt Supply Pressure: No paper + Low paper
    """
    features_created = []
    
    # Jam + Reader Failure interaction
    if 'card_reader_jammed' in df.columns and 'card_reader_state' in df.columns:
        df['jam_with_reader_failing'] = (
            (df['card_reader_jammed'] > 0) & (df['card_reader_state'] >= 1)
        ).astype(int)
        features_created.append('jam_with_reader_failing')
    
    # Cassette supply pressure (empty + low)
    if 'total_cassettes_empty' in df.columns and 'total_cassettes_low' in df.columns:
        df['cassette_supply_pressure'] = (
            df['total_cassettes_empty'].fillna(0) * 2 + 
            df['total_cassettes_low'].fillna(0)
        )
        features_created.append('cassette_supply_pressure')
    
    # Receipt supply pressure (out + low)
    if 'receipt_paper_out' in df.columns and 'receipt_paper_low' in df.columns:
        df['receipt_supply_pressure'] = (
            df['receipt_paper_out'].fillna(0) * 2 + 
            df['receipt_paper_low'].fillna(0)
        )
        features_created.append('receipt_supply_pressure')
    
    return df, features_created


def engineer_all_features(df, is_training=True):
    """
    Apply all feature engineering transformations.
    
    Parameters:
      df: Input DataFrame
      is_training: If True, create criticality scores; if False, assume they exist
    
    Returns:
      df: DataFrame with all 249 features engineered
    """
    # Start with criticality scores
    if is_training or 'criticality_score' not in df.columns:
        df['criticality_score'] = compute_criticality_score(df)
    
    if is_training or 'critical_incident_score' not in df.columns:
        df['critical_incident_score'] = compute_critical_incident_score(df)
    
    # Temporal features
    df = create_temporal_features(df)
    
    # Weekend-relative baselines
    df = create_weekend_relative_features(df)
    
    # Rolling windows
    df = create_rolling_features(df)
    
    # Cross-signals
    df, _ = create_interaction_features(df)
    
    # Fill remaining NaNs and infinities
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    
    return df


def get_numeric_features(df, exclude_cols=None):
    """
    Get list of numeric feature columns for modeling.
    
    Parameters:
      df: DataFrame
      exclude_cols: List of columns to exclude (e.g., 'atm_id', 'event_date')
    
    Returns:
      List of numeric column names suitable for ML models
    """
    if exclude_cols is None:
        exclude_cols = [
            'atm_id', 'event_date', 'has_failure_tomorrow',
            'max_state_tomorrow', 'max_failure_severity_tomorrow',
            'partial_failures_tomorrow', 'total_failures_tomorrow',
            'unknown_states_tomorrow', 'components_failed_tomorrow',
            'dispenser_partial_tomorrow', 'dispenser_total_tomorrow',
            'dispenser_unknown_tomorrow', 'card_reader_partial_tomorrow',
            'card_reader_total_tomorrow', 'card_reader_unknown_tomorrow',
            'host_comm_partial_tomorrow', 'host_comm_total_tomorrow',
            'host_comm_unknown_tomorrow', 'app_status_partial_tomorrow',
            'app_status_total_tomorrow', 'app_status_unknown_tomorrow',
            'device_type', 'device_style', 'vendor', 'model', 'branch'
        ]
    
    exclude_cols = [c.lower() for c in exclude_cols]
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    
    features = [c for c in numeric_cols if c not in exclude_cols]
    return features


if __name__ == '__main__':
    # Demo: Load sample data and engineer features
    print("Feature Engineering Module")
    print("=" * 70)
    print("\nUsage:")
    print("  from feature_engineering import engineer_all_features")
    print("  df = pd.read_csv('data.csv')")
    print("  df = engineer_all_features(df)")
    print("  features = get_numeric_features(df)")
    print("\nFeature Categories:")
    print("  1. Component State (21)")
    print("  2. Error Patterns (18)")
    print("  3. Temporal (12)")
    print("  4. Rolling Windows (40+)")
    print("  5. Baseline Ratios (15+)")
    print("  6. Cross-Signals (10+)")
    print("  =" * 70)
    print("  TOTAL: 249 Features")
