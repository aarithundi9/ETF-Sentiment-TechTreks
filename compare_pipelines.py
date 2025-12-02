"""
Compare mock vs real data pipeline performance.

This script demonstrates that the pipeline works with both data sources
and shows the performance difference.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.features.build_features import create_feature_pipeline, prepare_train_test_split
from src.models.train_model import train_baseline_model


def run_comparison():
    """Compare mock vs real data pipelines."""
    print("\n" + "=" * 80)
    print("MOCK vs REAL DATA COMPARISON")
    print("=" * 80)
    
    results = {}
    
    # Test 1: Mock Data
    print("\n[1/2] Testing MOCK DATA pipeline...")
    print("-" * 80)
    try:
        df_mock = create_feature_pipeline(use_mock_data=True, save_interim=False)
        X_train, X_test, y_train, y_test = prepare_train_test_split(df_mock)
        model_mock, metrics_mock = train_baseline_model(
            X_train, y_train, X_test, y_test,
            perform_cv=False,
            save_model=False,
        )
        results['mock'] = {
            'status': 'SUCCESS',
            'dataset_shape': df_mock.shape,
            'test_accuracy': metrics_mock['accuracy'],
            'test_f1': metrics_mock['f1_score'],
        }
        print(f"\n✓ Mock data pipeline: SUCCESS")
        print(f"  Dataset: {df_mock.shape}")
        print(f"  Accuracy: {metrics_mock['accuracy']:.4f}")
        print(f"  F1-Score: {metrics_mock['f1_score']:.4f}")
    except Exception as e:
        results['mock'] = {'status': 'FAILED', 'error': str(e)}
        print(f"\n❌ Mock data pipeline: FAILED")
        print(f"  Error: {e}")
    
    # Test 2: Real Data
    print("\n[2/2] Testing REAL DATA pipeline...")
    print("-" * 80)
    try:
        df_real = create_feature_pipeline(use_mock_data=False, save_interim=False)
        X_train, X_test, y_train, y_test = prepare_train_test_split(df_real)
        model_real, metrics_real = train_baseline_model(
            X_train, y_train, X_test, y_test,
            perform_cv=False,
            save_model=False,
        )
        results['real'] = {
            'status': 'SUCCESS',
            'dataset_shape': df_real.shape,
            'test_accuracy': metrics_real['accuracy'],
            'test_f1': metrics_real['f1_score'],
        }
        print(f"\n✓ Real data pipeline: SUCCESS")
        print(f"  Dataset: {df_real.shape}")
        print(f"  Accuracy: {metrics_real['accuracy']:.4f}")
        print(f"  F1-Score: {metrics_real['f1_score']:.4f}")
    except Exception as e:
        results['real'] = {'status': 'FAILED', 'error': str(e)}
        print(f"\n❌ Real data pipeline: FAILED")
        print(f"  Error: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if results['mock']['status'] == 'SUCCESS' and results['real']['status'] == 'SUCCESS':
        print("\n✓ Both pipelines working!")
        print("\nPerformance Comparison:")
        print(f"  {'Dataset':<12} {'Shape':<15} {'Accuracy':<12} {'F1-Score':<12}")
        print(f"  {'-'*12} {'-'*15} {'-'*12} {'-'*12}")
        print(f"  {'Mock Data':<12} {str(results['mock']['dataset_shape']):<15} "
              f"{results['mock']['test_accuracy']:<12.4f} {results['mock']['test_f1']:<12.4f}")
        print(f"  {'Real Data':<12} {str(results['real']['dataset_shape']):<15} "
              f"{results['real']['test_accuracy']:<12.4f} {results['real']['test_f1']:<12.4f}")
        
        print("\nNotes:")
        print("  - Mock data typically shows higher accuracy (synthetic correlations)")
        print("  - Real data is more challenging (actual market noise)")
        print("  - Both pipelines use identical feature engineering and modeling")
    else:
        print("\n⚠ Some pipelines failed. Check errors above.")
        if results['mock']['status'] == 'FAILED':
            print(f"  Mock: {results['mock']['error']}")
        if results['real']['status'] == 'FAILED':
            print(f"  Real: {results['real']['error']}")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_comparison()
