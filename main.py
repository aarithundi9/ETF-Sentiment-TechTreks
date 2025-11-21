"""
Main entry point for the ETF Sentiment Analysis project.

This script provides a CLI for running the complete pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.data.mock_data_generator import generate_all_mock_data
from src.features.build_features import create_feature_pipeline, prepare_train_test_split
from src.models.train_model import train_baseline_model
from src.config.settings import print_config


def generate_data(args):
    """Generate mock data."""
    print("\n" + "=" * 80)
    print("GENERATING MOCK DATA")
    print("=" * 80)
    
    prices_df, sentiment_df = generate_all_mock_data(save=True, verbose=True)
    
    print("\n✓ Mock data generation complete!")
    print(f"  Prices: {len(prices_df)} records")
    print(f"  Sentiment: {len(sentiment_df)} records")


def run_pipeline(args):
    """Run the complete feature engineering pipeline."""
    print("\n" + "=" * 80)
    print("RUNNING FEATURE ENGINEERING PIPELINE")
    print("=" * 80)
    
    df = create_feature_pipeline(
        use_mock_data=True,
        save_interim=True,
    )
    
    print(f"\n✓ Pipeline complete!")
    print(f"  Dataset shape: {df.shape}")
    print(f"  Output: data/processed/modeling_dataset.csv")


def train_model(args):
    """Train a model."""
    print("\n" + "=" * 80)
    print("TRAINING MODEL")
    print("=" * 80)
    
    # Run pipeline if needed
    print("\n[1/3] Running feature engineering...")
    df = create_feature_pipeline(use_mock_data=True, save_interim=True)
    
    # Prepare data
    print("\n[2/3] Preparing train/test split...")
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)
    
    # Train model
    print("\n[3/3] Training model...")
    model, metrics = train_baseline_model(
        X_train, y_train, X_test, y_test,
        perform_cv=True,
        save_model=True,
    )
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Test F1-Score: {metrics['f1_score']:.4f}")


def show_config(args):
    """Display current configuration."""
    print_config()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ETF Sentiment Analysis - ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py generate        # Generate mock data
  python main.py pipeline        # Run feature engineering
  python main.py train           # Train a model (includes pipeline)
  python main.py config          # Show configuration
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Generate data command
    parser_generate = subparsers.add_parser(
        'generate',
        help='Generate mock data'
    )
    parser_generate.set_defaults(func=generate_data)
    
    # Run pipeline command
    parser_pipeline = subparsers.add_parser(
        'pipeline',
        help='Run feature engineering pipeline'
    )
    parser_pipeline.set_defaults(func=run_pipeline)
    
    # Train model command
    parser_train = subparsers.add_parser(
        'train',
        help='Train a model'
    )
    parser_train.set_defaults(func=train_model)
    
    # Show config command
    parser_config = subparsers.add_parser(
        'config',
        help='Show configuration'
    )
    parser_config.set_defaults(func=show_config)
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Run the selected command
    try:
        args.func(args)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
