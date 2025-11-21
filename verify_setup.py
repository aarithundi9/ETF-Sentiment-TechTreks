"""
Project setup verification script.

Run this to check if your environment is properly configured.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def check_imports():
    """Check if all required packages are installed."""
    print("\n" + "=" * 80)
    print("CHECKING IMPORTS")
    print("=" * 80)
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'yaml': 'pyyaml',
        'requests': 'requests',
        'feedparser': 'feedparser',
        'bs4': 'beautifulsoup4',
        'yfinance': 'yfinance',
        'vaderSentiment': 'vaderSentiment',
        'sqlalchemy': 'SQLAlchemy',
        'pandas_ta': 'pandas-ta',
        'dotenv': 'python-dotenv',
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    else:
        print("\n✓ All required packages installed!")
        return True


def check_project_structure():
    """Check if project directories exist."""
    print("\n" + "=" * 80)
    print("CHECKING PROJECT STRUCTURE")
    print("=" * 80)
    
    required_dirs = [
        "data/raw",
        "data/interim",
        "data/processed",
        "notebooks",
        "src/config",
        "src/data",
        "src/features",
        "src/models",
        "tests",
    ]
    
    required_files = [
        "README.md",
        "requirements.txt",
        ".env.example",
        "main.py",
        "src/__init__.py",
        "src/config/settings.py",
        "src/data/mock_data_generator.py",
        "src/features/build_features.py",
        "src/models/train_model.py",
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ - MISSING")
            all_good = False
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_good = False
    
    if all_good:
        print("\n✓ Project structure is complete!")
    else:
        print("\n⚠ Some files/directories are missing!")
    
    return all_good


def check_config():
    """Check if configuration is loadable."""
    print("\n" + "=" * 80)
    print("CHECKING CONFIGURATION")
    print("=" * 80)
    
    try:
        from src.config.settings import (
            TICKERS, START_DATE, END_DATE, 
            RAW_DATA_DIR, PROCESSED_DATA_DIR
        )
        
        print(f"✓ Tickers: {TICKERS}")
        print(f"✓ Date range: {START_DATE} to {END_DATE}")
        print(f"✓ Raw data directory: {RAW_DATA_DIR}")
        print(f"✓ Processed data directory: {PROCESSED_DATA_DIR}")
        
        print("\n✓ Configuration loaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_mock_data_generation():
    """Test if mock data can be generated."""
    print("\n" + "=" * 80)
    print("TESTING MOCK DATA GENERATION")
    print("=" * 80)
    
    try:
        from src.data.mock_data_generator import generate_price_series
        
        df = generate_price_series(
            ticker="TEST",
            start_date="2023-01-01",
            end_date="2023-01-31",
            initial_price=100.0,
            seed=42,
        )
        
        print(f"✓ Generated {len(df)} price records")
        print(f"✓ Columns: {list(df.columns)}")
        print("\n✓ Mock data generation works!")
        return True
    except Exception as e:
        print(f"✗ Mock data generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_technical_indicators():
    """Test if technical indicators can be calculated."""
    print("\n" + "=" * 80)
    print("TESTING TECHNICAL INDICATORS")
    print("=" * 80)
    
    try:
        import pandas as pd
        import numpy as np
        from src.data.technical_data import calculate_sma, calculate_rsi
        
        # Create sample data
        df = pd.DataFrame({
            'date': pd.bdate_range('2023-01-01', periods=50),
            'ticker': 'TEST',
            'close': 100 + np.cumsum(np.random.randn(50)),
        })
        
        # Test SMA
        df = calculate_sma(df, [5, 10])
        assert 'sma_5' in df.columns
        print("✓ SMA calculation works")
        
        # Test RSI
        df = calculate_rsi(df)
        assert 'rsi' in df.columns
        print("✓ RSI calculation works")
        
        print("\n✓ Technical indicators work!")
        return True
    except Exception as e:
        print(f"✗ Technical indicators failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_checks():
    """Run all verification checks."""
    print("\n" + "=" * 80)
    print("ETF SENTIMENT ANALYSIS - PROJECT VERIFICATION")
    print("=" * 80)
    print(f"Python version: {sys.version}")
    print(f"Project root: {project_root}")
    
    results = {
        "Imports": check_imports(),
        "Project Structure": check_project_structure(),
        "Configuration": check_config(),
        "Mock Data Generation": test_mock_data_generation(),
        "Technical Indicators": test_technical_indicators(),
    }
    
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check:.<40} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL CHECKS PASSED! Your environment is ready!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. python main.py generate    # Generate mock data")
        print("  2. python main.py train       # Train a model")
        print("  3. jupyter notebook           # Explore in Jupyter")
    else:
        print("⚠ SOME CHECKS FAILED! Please fix the issues above.")
        print("=" * 80)
        print("\nTroubleshooting:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Check file structure matches PROJECT_OVERVIEW.md")
        print("  - Review error messages above")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
