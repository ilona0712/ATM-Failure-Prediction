"""
ATM Failure Prediction - Daily Predictions (v2.1)
==================================================
Generates daily predictions for all ATMs using trained model.

Input:  today_data.csv (15-day history) + best_model.pkl
Output: daily_predictions_YYYY-MM-DD.csv

Purpose:
  - Load trained XGBoost model
  - Engineer features from 15-day window
  - Generate calibrated probabilities
  - Assign risk tiers
  - Detect correlated infrastructure events
  - Output ranked watchlist for operations team
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timezone, timedelta
import warnings

warnings.filterwarnings('ignore')

# Try to import feature engineering module
try:
    from feature_engineering import engineer_all_features, get_numeric_features
except ImportError:
    print("Warning: feature_engineering module not found in path")

# Timezone support (Lebanon EET/EEST)
LEBANON_TZ = timezone(timedelta(hours=2))

def get_lebanon_time():
    """Get current time in Lebanon timezone"""
    return datetime.now(LEBANON_TZ).strftime('%Y-%m-%d %H:%M:%S') + " EET"

def get_lebanon_date():
    """Get current date in Lebanon timezone"""
    return datetime.now(LEBANON_TZ).strftime('%Y-%m-%d')


class DailyPredictor:
    """Generate daily predictions for ATM failure risk."""
    
    def __init__(self, model_path='models/best_model.pkl'):
        """
        Initialize predictor with trained model.
        
        Parameters:
            model_path: Path to best_model.pkl file
        """
        print(f"\n{'='*70}")
        print("ATM FAILURE PREDICTION - DAILY PREDICTIONS v2.1")
        print(f"Started: {get_lebanon_time()}")
        print(f"{'='*70}")
        
        # Load model package
        print("\nSTEP 1: LOADING MODEL")
        print(f"{'='*70}")
        
        try:
            self.model_pkg = joblib.load(model_path)
            self.model = self.model_pkg['model']
            self.calibrator = self.model_pkg.get('calibrator', None)
            self.features = self.model_pkg['feature_names']
            
            print(f"   Model: {self.model_pkg.get('model_name', 'Unknown')}")
            print(f"   Features: {len(self.features)}")
            print(f"   Calibrator: {'Isotonic Regression' if self.calibrator else 'None'}")
            print(f"   Version: {self.model_pkg.get('version', 'Unknown')}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def load_data(self, data_path):
        """
        Load ATM operational data.
        
        Parameters:
            data_path: Path to input CSV file (15-day window)
        
        Returns:
            DataFrame with raw operational data
        """
        print("\nSTEP 2: LOADING DATA")
        print(f"{'='*70}")
        
        # Try different separators
        df = None
        for sep in [',', '\t', None]:
            try:
                kw = {} if sep is not None else {'engine': 'python'}
                df = pd.read_csv(data_path, sep=sep, **kw)
                if len(df.columns) > 5:
                    break
            except Exception:
                continue
        
        if df is None or len(df.columns) <= 5:
            raise ValueError(f"Could not load file: {data_path}")
        
        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()
        
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Date range: {df['event_date'].min()} to {df['event_date'].max()}")
        
        return df
    
    def engineer_features(self, df):
        """
        Engineer all 249 features from raw data.
        
        Parameters:
            df: Raw operational data
        
        Returns:
            DataFrame with engineered features
        """
        print("\nSTEP 3: ENGINEERING FEATURES")
        print(f"{'='*70}")
        
        # Engineer features
        df = engineer_all_features(df, is_training=False)
        
        print(f"   Features created: {len(df.columns)}")
        
        return df
    
    def generate_predictions(self, df):
        """
        Generate raw probabilities from model.
        
        Parameters:
            df: DataFrame with engineered features
        
        Returns:
            Array of raw probabilities (0.0 to 1.0)
        """
        print("\nSTEP 4: GENERATING PREDICTIONS")
        print(f"{'='*70}")
        
        # Extract features in correct order
        X = df[self.features].fillna(0).replace([np.inf, -np.inf], 0)
        
        print(f"   Input shape: {X.shape}")
        
        # Generate raw probabilities
        raw_probs = self.model.predict_proba(X)[:, 1]
        
        print(f"   Predictions generated: {len(raw_probs)}")
        print(f"   Mean probability: {raw_probs.mean():.4f}")
        print(f"   Min-Max: {raw_probs.min():.4f} - {raw_probs.max():.4f}")
        
        return raw_probs
    
    def calibrate_probabilities(self, raw_probs):
        """
        Apply isotonic calibration to raw probabilities.
        
        Parameters:
            raw_probs: Raw model probabilities
        
        Returns:
            Calibrated probabilities
        """
        print("\nSTEP 5: CALIBRATING PROBABILITIES")
        print(f"{'='*70}")
        
        if self.calibrator is None:
            print("   No calibrator found, using raw probabilities")
            return raw_probs
        
        # Apply calibration
        cal_probs = self.calibrator.predict(raw_probs)
        
        print(f"   Mean probability (raw): {raw_probs.mean():.4f}")
        print(f"   Mean probability (cal): {cal_probs.mean():.4f}")
        print(f"   Calibration applied: Isotonic Regression")
        
        return cal_probs
    
    def assign_risk_tiers(self, probs):
        """
        Assign risk tiers based on calibrated probabilities.
        
        Parameters:
            probs: Calibrated probabilities
        
        Returns:
            Array of risk tier strings
        """
        # Define thresholds
        CRITICAL_THRESHOLD = 0.65
        HIGH_THRESHOLD = 0.20
        MEDIUM_THRESHOLD = 0.10
        
        tiers = np.where(
            probs >= CRITICAL_THRESHOLD, 'CRITICAL',
            np.where(
                probs >= HIGH_THRESHOLD, 'HIGH',
                np.where(
                    probs >= MEDIUM_THRESHOLD, 'MEDIUM', 'LOW'
                )
            )
        )
        
        # Count by tier
        unique, counts = np.unique(tiers, return_counts=True)
        print("\nSTEP 6: ASSIGNING RISK TIERS")
        print(f"{'='*70}")
        for tier, count in zip(unique, counts):
            print(f"   {tier:>10}: {count:>4} ATMs")
        
        return tiers
    
    def detect_correlated_events(self, tiers):
        """
        Detect fleet-wide infrastructure events.
        
        If >50% of ATMs flagged HIGH/CRITICAL, it's likely an infrastructure
        outage rather than individual ATM failures.
        
        Parameters:
            tiers: Array of risk tiers
        
        Returns:
            Boolean indicating if correlated event detected
        """
        print("\nSTEP 7: DETECTING CORRELATED EVENTS")
        print(f"{'='*70}")
        
        critical_or_high = np.sum((tiers == 'CRITICAL') | (tiers == 'HIGH'))
        total = len(tiers)
        pct = critical_or_high / total
        
        is_correlated = pct > 0.50
        
        print(f"   High/Critical ATMs: {critical_or_high}/{total} ({pct:.1%})")
        print(f"   Correlated event: {'YES ⚠️' if is_correlated else 'NO ✓'}")
        
        if is_correlated:
            print(f"   Action: Alert network team (infrastructure issue)")
        else:
            print(f"   Action: Individual dispatch routing")
        
        return is_correlated
    
    def create_output(self, df, probs, tiers, is_correlated):
        """
        Create output DataFrame with predictions and rankings.
        
        Parameters:
            df: Original data with ATM IDs and branch info
            probs: Calibrated probabilities
            tiers: Risk tier assignments
            is_correlated: Whether infrastructure event detected
        
        Returns:
            Output DataFrame ready for export
        """
        print("\nSTEP 8: CREATING OUTPUT")
        print(f"{'='*70}")
        
        # Create output dataframe
        results = pd.DataFrame({
            'atm_id': df['atm_id'],
            'failure_probability': probs.round(4),
            'risk_level': tiers,
            'criticality_score': df.get('criticality_score', 0).fillna(0).astype(int),
            'critical_incident_score': df.get('critical_incident_score', 0).fillna(0).astype(int),
            'branch': df.get('branch', 'Unknown'),
            'is_infrastructure_event': is_correlated,
        })
        
        # Sort by probability (highest first)
        results = results.sort_values('failure_probability', ascending=False).reset_index(drop=True)
        results['rank'] = range(1, len(results) + 1)
        
        # Reorder columns
        results = results[[
            'rank', 'atm_id', 'branch', 'failure_probability', 'risk_level',
            'criticality_score', 'critical_incident_score', 'is_infrastructure_event'
        ]]
        
        print(f"   Output rows: {len(results)}")
        print(f"   Columns: {len(results.columns)}")
        print(f"   Top 5 highest risk:")
        for i, row in results.head(5).iterrows():
            print(f"      {row['rank']}. {row['atm_id']:>12} {row['branch']:>20} "
                  f"{row['failure_probability']:.3f} {row['risk_level']}")
        
        return results
    
    def save_output(self, results, output_dir='predictions'):
        """
        Save predictions to CSV file.
        
        Parameters:
            results: Output DataFrame
            output_dir: Directory to save predictions
        """
        print("\nSTEP 9: SAVING OUTPUT")
        print(f"{'='*70}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename with date
        filename = f"daily_predictions_{get_lebanon_date()}.csv"
        filepath = output_path / filename
        
        # Save
        results.to_csv(filepath, index=False)
        
        print(f"   Saved: {filepath}")
        print(f"   Size: {len(results):,} rows")
        
        return filepath
    
    def predict(self, data_path, output_dir='predictions'):
        """
        Complete prediction pipeline.
        
        Parameters:
            data_path: Path to input data CSV
            output_dir: Directory for output predictions
        
        Returns:
            Output DataFrame with predictions
        """
        # Load and engineer features
        df = self.load_data(data_path)
        df_engineered = self.engineer_features(df)
        
        # Generate and calibrate predictions
        raw_probs = self.generate_predictions(df_engineered)
        cal_probs = self.calibrate_probabilities(raw_probs)
        
        # Assign tiers and detect events
        tiers = self.assign_risk_tiers(cal_probs)
        is_correlated = self.detect_correlated_events(tiers)
        
        # Create and save output
        results = self.create_output(df, cal_probs, tiers, is_correlated)
        self.save_output(results, output_dir)
        
        print("\n" + "="*70)
        print("PREDICTIONS COMPLETE!")
        print(f"Finished: {get_lebanon_time()}")
        print("="*70 + "\n")
        
        return results


def main():
    """Main entry point for daily predictions."""
    import sys
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python daily_predictions.py <input_data_path> [output_dir]")
        print("  input_data_path: Path to today_data.csv (15-day window)")
        print("  output_dir: Directory for predictions (default: ./predictions)")
        sys.exit(1)
    
    data_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'predictions'
    
    # Run predictions
    try:
        predictor = DailyPredictor()
        results = predictor.predict(data_path, output_dir)
        print("\n✓ Success! Predictions generated and saved.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
