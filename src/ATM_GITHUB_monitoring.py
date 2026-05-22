"""
ATM Failure Prediction - Daily Monitoring (v2.1)
=================================================
Evaluates daily predictions against actual failures.

Input:  daily_predictions_YYYY-MM-DD.csv + actual_failures_YYYY-MM-DD.csv
Output: monitoring_log.csv (rolling history) + monitoring_report_YYYY-MM-DD.txt

Purpose:
  - Compare predictions to actual failures
  - Calculate performance metrics (PR-AUC, ROC-AUC, recall, precision, F1)
  - Track model drift
  - Generate alerts if metrics degrade
  - Detect new/unknown ATMs and branches
  - Log everything for trend analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score, f1_score,
    confusion_matrix, precision_score, recall_score
)
import warnings

warnings.filterwarnings('ignore')

# Timezone support (Lebanon EET/EEST)
LEBANON_TZ = timezone(timedelta(hours=2))

def get_lebanon_time():
    """Get current time in Lebanon timezone"""
    return datetime.now(LEBANON_TZ).strftime('%Y-%m-%d %H:%M:%S') + " EET"

def get_lebanon_date():
    """Get current date in Lebanon timezone"""
    return datetime.now(LEBANON_TZ).strftime('%Y-%m-%d')


class ATMMonitor:
    """Monitor daily prediction accuracy and model drift."""
    
    def __init__(self, log_path='monitoring_log.csv', report_dir='reports'):
        """
        Initialize monitoring system.
        
        Parameters:
            log_path: Path to rolling monitoring log
            report_dir: Directory for daily reports
        """
        self.log_path = Path(log_path)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print("ATM FAILURE PREDICTION - DAILY MONITORING v2.1")
        print(f"Started: {get_lebanon_time()}")
        print(f"{'='*70}")
        
        # Load existing log
        if self.log_path.exists():
            self.log = pd.read_csv(self.log_path)
            print(f"\nLoaded existing log: {len(self.log)} days recorded")
        else:
            self.log = None
            print(f"\nNo existing log found, creating new")
    
    def load_predictions(self, predictions_file):
        """Load yesterday's predictions."""
        print("\nSTEP 1: LOADING PREDICTIONS")
        print(f"{'='*70}")
        
        predictions = pd.read_csv(predictions_file)
        print(f"   Predictions loaded: {len(predictions)} ATMs")
        
        # Extract probability column
        if 'failure_probability' in predictions.columns:
            probs = predictions['failure_probability'].values
        elif 'probability' in predictions.columns:
            probs = predictions['probability'].values
        else:
            raise ValueError("No probability column found in predictions")
        
        return predictions, probs
    
    def load_actuals(self, actuals_file):
        """Load actual failures from yesterday."""
        print("\nSTEP 2: LOADING ACTUALS")
        print(f"{'='*70}")
        
        actuals = pd.read_csv(actuals_file)
        print(f"   Actuals loaded: {len(actuals)} ATMs")
        
        # Extract failure column
        if 'has_failure' in actuals.columns:
            failures = actuals['has_failure'].astype(int).values
        elif 'failure' in actuals.columns:
            failures = actuals['failure'].astype(int).values
        else:
            raise ValueError("No failure column found in actuals")
        
        actual_failure_count = failures.sum()
        actual_failure_rate = failures.mean()
        
        print(f"   Actual failures: {actual_failure_count}/{len(failures)} ({actual_failure_rate:.1%})")
        
        return actuals, failures
    
    def detect_infrastructure_event(self, failures):
        """
        Detect if >70% of fleet actually failed (infrastructure event).
        
        Parameters:
            failures: Array of actual failure indicators
        
        Returns:
            Boolean indicating if infrastructure event
        """
        print("\nSTEP 3: DETECTING INFRASTRUCTURE EVENTS")
        print(f"{'='*70}")
        
        failure_rate = failures.mean()
        is_infrastructure_event = failure_rate > 0.70
        
        print(f"   Fleet failure rate: {failure_rate:.1%}")
        print(f"   Infrastructure event: {'YES ⚠️' if is_infrastructure_event else 'NO ✓'}")
        
        if is_infrastructure_event:
            print(f"   Note: Per-ATM metrics unreliable for this date")
        
        return is_infrastructure_event
    
    def calculate_metrics(self, probs, failures):
        """
        Calculate comprehensive evaluation metrics.
        
        Parameters:
            probs: Predicted probabilities
            failures: Actual failure indicators
        
        Returns:
            Dictionary of metrics
        """
        print("\nSTEP 4: CALCULATING METRICS")
        print(f"{'='*70}")
        
        # Precision-Recall curve and AUC
        precision, recall, _ = precision_recall_curve(failures, probs)
        pr_auc = auc(recall, precision)
        
        # ROC-AUC
        roc_auc = roc_auc_score(failures, probs)
        
        # Binary predictions (threshold 0.5)
        pred_binary = (probs >= 0.5).astype(int)
        
        # Metrics
        precision_score_val = precision_score(failures, pred_binary, zero_division=0)
        recall_score_val = recall_score(failures, pred_binary, zero_division=0)
        f1_score_val = f1_score(failures, pred_binary, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(failures, pred_binary).ravel()
        
        metrics = {
            'date': get_lebanon_date(),
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
            'precision': precision_score_val,
            'recall': recall_score_val,
            'f1': f1_score_val,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'total_atms': len(failures),
            'actual_failures': int(failures.sum()),
        }
        
        print(f"   PR-AUC:   {pr_auc:.4f}")
        print(f"   ROC-AUC:  {roc_auc:.4f}")
        print(f"   Recall:   {recall_score_val:.4f}")
        print(f"   Precision: {precision_score_val:.4f}")
        print(f"   F1:       {f1_score_val:.4f}")
        print(f"   TP: {tp:,}  FP: {fp:,}  FN: {fn:,}  TN: {tn:,}")
        
        return metrics
    
    def check_alerts(self, metrics):
        """
        Check alert conditions.
        
        Alert Thresholds:
          - PR-AUC < 0.65
          - Recall < 0.50
          - Precision < 0.40
        
        Parameters:
            metrics: Dictionary of calculated metrics
        
        Returns:
            List of alert messages
        """
        print("\nSTEP 5: CHECKING ALERTS")
        print(f"{'='*70}")
        
        alerts = []
        
        # PR-AUC floor
        if metrics['pr_auc'] < 0.65:
            alerts.append(f"PR-AUC {metrics['pr_auc']:.4f} < 0.65 (floor)")
        
        # Recall floor
        if metrics['recall'] < 0.50:
            alerts.append(f"Recall {metrics['recall']:.4f} < 0.50 (floor)")
        
        # Precision floor
        if metrics['precision'] < 0.40:
            alerts.append(f"Precision {metrics['precision']:.4f} < 0.40 (floor)")
        
        if alerts:
            print(f"   ⚠️  ALERTS TRIGGERED:")
            for alert in alerts:
                print(f"       - {alert}")
        else:
            print(f"   ✓ All metrics within acceptable ranges")
        
        return alerts
    
    def update_monitoring_log(self, metrics):
        """
        Update rolling monitoring log.
        
        Parameters:
            metrics: Dictionary of metrics for today
        """
        print("\nSTEP 6: UPDATING LOG")
        print(f"{'='*70}")
        
        # Create new log entry
        new_entry = pd.DataFrame([metrics])
        
        # Append or create log
        if self.log is None:
            self.log = new_entry
        else:
            self.log = pd.concat([self.log, new_entry], ignore_index=True)
        
        # Save log
        self.log.to_csv(self.log_path, index=False)
        print(f"   Log updated: {len(self.log)} days recorded")
        print(f"   Saved to: {self.log_path}")
        
        # Calculate 5-day rolling average
        if len(self.log) >= 5:
            rolling_5d = self.log.tail(5)[['pr_auc', 'recall', 'precision']].mean()
            print(f"\n   5-day rolling average:")
            print(f"      PR-AUC:   {rolling_5d['pr_auc']:.4f}")
            print(f"      Recall:   {rolling_5d['recall']:.4f}")
            print(f"      Precision: {rolling_5d['precision']:.4f}")
    
    def generate_report(self, metrics, alerts, predictions, actuals):
        """
        Generate human-readable daily report.
        
        Parameters:
            metrics: Dictionary of metrics
            alerts: List of alert messages
            predictions: Predictions DataFrame
            actuals: Actuals DataFrame
        """
        print("\nSTEP 7: GENERATING REPORT")
        print(f"{'='*70}")
        
        report_file = self.report_dir / f"monitoring_report_{get_lebanon_date()}.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("ATM FAILURE PREDICTION - DAILY EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Date: {metrics['date']}\n")
            f.write(f"Generated: {get_lebanon_time()}\n\n")
            
            f.write("SUMMARY\n")
            f.write("-"*70 + "\n")
            f.write(f"ATMs Evaluated: {metrics['total_atms']}\n")
            f.write(f"Actual Failures: {metrics['actual_failures']}/{metrics['total_atms']} "
                   f"({metrics['actual_failures']/metrics['total_atms']:.1%})\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-"*70 + "\n")
            f.write(f"PR-AUC:    {metrics['pr_auc']:.4f} (floor: 0.65) "
                   f"{'✓' if metrics['pr_auc'] >= 0.65 else '✗'}\n")
            f.write(f"ROC-AUC:   {metrics['roc_auc']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f} (floor: 0.50) "
                   f"{'✓' if metrics['recall'] >= 0.50 else '✗'}\n")
            f.write(f"Precision: {metrics['precision']:.4f} (floor: 0.40) "
                   f"{'✓' if metrics['precision'] >= 0.40 else '✗'}\n")
            f.write(f"F1 Score:  {metrics['f1']:.4f}\n\n")
            
            f.write("CONFUSION MATRIX\n")
            f.write("-"*70 + "\n")
            f.write(f"True Positives:  {metrics['tp']:>5}  (caught failures)\n")
            f.write(f"False Positives: {metrics['fp']:>5}  (unnecessary dispatch)\n")
            f.write(f"False Negatives: {metrics['fn']:>5}  (missed failures)\n")
            f.write(f"True Negatives:  {metrics['tn']:>5}  (correct no-failure)\n\n")
            
            # Tier performance
            f.write("TIER PERFORMANCE\n")
            f.write("-"*70 + "\n")
            
            for tier in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                if 'risk_level' in predictions.columns:
                    pred_in_tier = predictions['risk_level'] == tier
                    if pred_in_tier.sum() > 0:
                        # Find corresponding actuals
                        actual_failures_in_tier = actuals[pred_in_tier][
                            'has_failure' if 'has_failure' in actuals.columns else 'failure'
                        ].sum()
                        actual_failure_rate = actual_failures_in_tier / pred_in_tier.sum()
                        
                        f.write(f"{tier:>10}: {pred_in_tier.sum():>4} flagged, "
                               f"{actual_failures_in_tier:>3} failed "
                               f"({actual_failure_rate:.1%})\n")
            
            f.write("\n")
            
            # Alerts
            f.write("ALERTS\n")
            f.write("-"*70 + "\n")
            if alerts:
                for alert in alerts:
                    f.write(f"⚠️  {alert}\n")
            else:
                f.write("✓ No alerts\n")
            
            f.write("\n")
            
            f.write("="*70 + "\n")
            f.write(f"Report generated at {get_lebanon_time()}\n")
            f.write("="*70 + "\n")
        
        print(f"   Report saved: {report_file}")
    
    def monitor(self, predictions_file, actuals_file):
        """
        Complete daily monitoring pipeline.
        
        Parameters:
            predictions_file: Path to daily_predictions_YYYY-MM-DD.csv
            actuals_file: Path to actual failures CSV
        
        Returns:
            Dictionary with evaluation results
        """
        # Load data
        predictions, probs = self.load_predictions(predictions_file)
        actuals, failures = self.load_actuals(actuals_file)
        
        # Detect infrastructure event
        is_infrastructure_event = self.detect_infrastructure_event(failures)
        
        # Calculate metrics
        metrics = self.calculate_metrics(probs, failures)
        metrics['is_infrastructure_event'] = is_infrastructure_event
        
        # Check alerts
        alerts = self.check_alerts(metrics)
        
        # Update log
        self.update_monitoring_log(metrics)
        
        # Generate report
        self.generate_report(metrics, alerts, predictions, actuals)
        
        print("\n" + "="*70)
        print("MONITORING COMPLETE!")
        print(f"Finished: {get_lebanon_time()}")
        print("="*70 + "\n")
        
        return {
            'metrics': metrics,
            'alerts': alerts,
            'is_infrastructure_event': is_infrastructure_event,
        }


def main():
    """Main entry point for daily monitoring."""
    import sys
    
    # Parse arguments
    if len(sys.argv) < 3:
        print("Usage: python monitoring.py <predictions_file> <actuals_file> [log_path]")
        print("  predictions_file: Path to daily_predictions_YYYY-MM-DD.csv")
        print("  actuals_file: Path to actual_failures_YYYY-MM-DD.csv")
        print("  log_path: Path to monitoring_log.csv (default: ./monitoring_log.csv)")
        sys.exit(1)
    
    predictions_file = sys.argv[1]
    actuals_file = sys.argv[2]
    log_path = sys.argv[3] if len(sys.argv) > 3 else 'monitoring_log.csv'
    
    # Run monitoring
    try:
        monitor = ATMMonitor(log_path=log_path)
        results = monitor.monitor(predictions_file, actuals_file)
        print("\n✓ Success! Monitoring complete.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
