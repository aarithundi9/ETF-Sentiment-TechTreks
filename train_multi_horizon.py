"""
Multi-Horizon ETF Price Prediction Training Script.

This script trains a multi-horizon LSTM model for ETF price prediction.
It supports training on single or multiple tickers with 5-day and 1-month
ahead prediction horizons.

Usage:
    # Train on single ticker
    python train_multi_horizon.py --ticker QQQ
    
    # Train on multiple tickers
    python train_multi_horizon.py --ticker QQQ XLY XLI XLF
    
    # Train on all available tickers
    python train_multi_horizon.py --all
    
    # Custom epochs and batch size
    python train_multi_horizon.py --ticker QQQ --epochs 200 --batch-size 64
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.settings import (
    MULTI_HORIZON_CONFIG,
    PROCESSED_DATA_DIR,
    TICKERS,
)
from src.features.build_features import (
    make_multi_horizon_targets,
    prepare_multi_horizon_sequences,
    load_ticker_dataset,
)
from src.models.multi_horizon_model import (
    MultiHorizonLSTM,
    MultiHorizonTrainer,
    create_data_loaders,
    get_default_feature_columns,
)
from src.models.evaluate_model import (
    evaluate_multi_horizon,
    create_multi_horizon_report,
    plot_multi_horizon_predictions,
)


def get_available_tickers() -> list:
    """Get list of tickers with available datasets."""
    available = []
    for ticker in ['QQQ', 'XLY', 'XLI', 'XLF', 'SPY', 'XLK', 'XLC']:
        filepath = PROCESSED_DATA_DIR / f"{ticker}total.csv"
        if filepath.exists():
            available.append(ticker)
    return available


def prepare_features_for_model(df: pd.DataFrame) -> list:
    """
    Get feature columns that exist in the dataframe.
    
    Returns filtered list of feature columns based on what's available.
    """
    # Get default features
    default_features = get_default_feature_columns()
    
    # Filter to only columns that exist
    available_features = [col for col in default_features if col in df.columns]
    
    # Add any additional numeric columns that might be useful
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude target columns and date
    exclude = ['target', 'target_5d', 'target_1m', 'forward_return', 'date']
    additional = [col for col in numeric_cols if col not in available_features 
                  and col not in exclude and not col.startswith('target_')]
    
    # Use available + any additional (but cap to avoid too many features)
    all_features = available_features + additional
    
    print(f"  Using {len(all_features)} features")
    
    return all_features


def train_single_ticker(
    ticker: str,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    sequence_length: int = None,
    save_model: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Train multi-horizon model for a single ticker.
    
    Args:
        ticker: Ticker symbol
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Optimizer learning rate
        sequence_length: LSTM sequence length
        save_model: Whether to save the trained model
        verbose: Print training progress
        
    Returns:
        Dictionary with training results and metrics
    """
    config = MULTI_HORIZON_CONFIG
    
    # Use config defaults if not specified
    epochs = epochs or config["training"]["epochs"]
    batch_size = batch_size or config["training"]["batch_size"]
    learning_rate = learning_rate or config["training"]["learning_rate"]
    sequence_length = sequence_length or config["model"]["sequence_length"]
    
    print(f"\n{'='*60}")
    print(f"Training Multi-Horizon Model: {ticker}")
    print(f"{'='*60}")
    
    # Load dataset
    try:
        df = load_ticker_dataset(ticker)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return {"ticker": ticker, "status": "failed", "error": str(e)}
    
    # Add multi-horizon targets
    print("\nCreating multi-horizon targets...")
    df = make_multi_horizon_targets(
        df,
        horizons=config["horizons"],
        target_type=config["target_type"],
    )
    
    # Get feature columns
    feature_cols = prepare_features_for_model(df)
    target_cols = [f"target_{h}" for h in config["horizons"].keys()]
    
    print(f"  Horizons: {list(config['horizons'].keys())}")
    print(f"  Target columns: {target_cols}")
    
    # Prepare sequences
    print("\nPreparing sequences...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_multi_horizon_sequences(
        df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        sequence_length=sequence_length,
        split_ratio=1 - config["val_split"],  # train+val vs test
        val_ratio=config["val_split"],
    )
    
    # Check for valid data
    if len(X_train) < batch_size:
        print(f"ERROR: Not enough training samples ({len(X_train)}) for batch size {batch_size}")
        return {"ticker": ticker, "status": "failed", "error": "Insufficient data"}
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X_train, X_val, y_train, y_val, batch_size=batch_size
    )
    
    # Create test loader for evaluation
    from torch.utils.data import DataLoader
    from src.models.multi_horizon_model import TimeSeriesDataset
    test_dataset = TimeSeriesDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    input_size = X_train.shape[2]  # n_features
    model = MultiHorizonLSTM.from_config(input_size=input_size, config=config["model"])
    
    print(f"\nModel Architecture:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {model.hidden_size}")
    print(f"  Num layers: {model.num_layers}")
    print(f"  Num horizons: {model.n_horizons}")
    
    # Create trainer
    checkpoint_dir = PROCESSED_DATA_DIR / "checkpoints" / ticker
    trainer = MultiHorizonTrainer(
        model=model,
        loss_weights=config["loss_weights"],
        learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir,
    )
    
    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping_patience=config["training"]["early_stopping_patience"],
        verbose=verbose,
    )
    
    # Evaluate on test set
    print("\n" + "-" * 40)
    print("Evaluating on test set...")
    
    y_pred = trainer.predict(X_test)
    
    test_metrics = evaluate_multi_horizon(
        y_true=y_test,
        y_pred=y_pred,
        horizon_names=list(config["horizons"].keys()),
        verbose=True,
    )
    
    # Save model if requested
    if save_model:
        final_model_path = checkpoint_dir / f"{ticker}_multi_horizon_final.pt"
        trainer.save_checkpoint(f"{ticker}_multi_horizon_final.pt")
        print(f"\nModel saved to: {final_model_path}")
    
    # Create report
    report = create_multi_horizon_report(
        metrics=test_metrics,
        ticker=ticker,
        save_path=checkpoint_dir / f"{ticker}_report.txt",
    )
    
    # Try to plot (may fail if no display)
    try:
        plot_path = checkpoint_dir / f"{ticker}_predictions.png"
        plot_multi_horizon_predictions(
            y_true=y_test,
            y_pred=y_pred,
            horizon_names=list(config["horizons"].keys()),
            title=f"{ticker} Multi-Horizon Predictions",
            save_path=str(plot_path),
        )
    except Exception as e:
        print(f"Could not create plot: {e}")
    
    return {
        "ticker": ticker,
        "status": "success",
        "history": history,
        "test_metrics": test_metrics,
        "model_path": str(checkpoint_dir / f"{ticker}_multi_horizon_final.pt"),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_features": input_size,
    }


def train_multiple_tickers(
    tickers: list,
    **kwargs,
) -> dict:
    """
    Train multi-horizon model for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        **kwargs: Arguments passed to train_single_ticker
        
    Returns:
        Dictionary with results for each ticker
    """
    results = {}
    
    print("\n" + "=" * 70)
    print(f"Training Multi-Horizon Models for {len(tickers)} Tickers")
    print("=" * 70)
    print(f"Tickers: {', '.join(tickers)}")
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
        results[ticker] = train_single_ticker(ticker, **kwargs)
    
    # Summary
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    
    successful = [t for t, r in results.items() if r.get("status") == "success"]
    failed = [t for t, r in results.items() if r.get("status") == "failed"]
    
    print(f"\nSuccessful: {len(successful)}/{len(tickers)}")
    if successful:
        print(f"  Tickers: {', '.join(successful)}")
        
        # Aggregate metrics
        print("\nTest Set Performance Summary:")
        for ticker in successful:
            metrics = results[ticker]["test_metrics"]
            print(f"\n  {ticker}:")
            for horizon, m in metrics["horizons"].items():
                print(f"    {horizon}: MAE={m['mae']:.6f}, Dir.Acc={m['directional_accuracy']:.2%}")
    
    if failed:
        print(f"\nFailed: {len(failed)}")
        for ticker in failed:
            print(f"  {ticker}: {results[ticker].get('error', 'Unknown error')}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Multi-Horizon LSTM for ETF Price Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_multi_horizon.py --ticker QQQ
  python train_multi_horizon.py --ticker QQQ XLY XLI XLF
  python train_multi_horizon.py --all
  python train_multi_horizon.py --ticker QQQ --epochs 200 --batch-size 64
        """
    )
    
    parser.add_argument(
        "--ticker", "-t",
        nargs="+",
        help="Ticker symbol(s) to train on",
    )
    
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Train on all available tickers",
    )
    
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=None,
        help=f"Number of training epochs (default: {MULTI_HORIZON_CONFIG['training']['epochs']})",
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=None,
        help=f"Batch size (default: {MULTI_HORIZON_CONFIG['training']['batch_size']})",
    )
    
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=None,
        help=f"Learning rate (default: {MULTI_HORIZON_CONFIG['training']['learning_rate']})",
    )
    
    parser.add_argument(
        "--sequence-length", "-s",
        type=int,
        default=None,
        help=f"Sequence length (default: {MULTI_HORIZON_CONFIG['model']['sequence_length']})",
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save trained model",
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output",
    )
    
    args = parser.parse_args()
    
    # Determine tickers to train
    if args.all:
        tickers = get_available_tickers()
        if not tickers:
            print("ERROR: No datasets found in data/processed/")
            print("Run create_ticker_dataset.py first to create datasets.")
            sys.exit(1)
    elif args.ticker:
        tickers = args.ticker
    else:
        # Default to QQQ if available
        available = get_available_tickers()
        if 'QQQ' in available:
            tickers = ['QQQ']
        elif available:
            tickers = [available[0]]
        else:
            print("ERROR: No datasets found. Specify ticker with --ticker or run create_ticker_dataset.py")
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print("Multi-Horizon ETF Price Prediction Training")
    print(f"{'='*60}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Horizons: {list(MULTI_HORIZON_CONFIG['horizons'].keys())}")
    
    # Training arguments
    train_kwargs = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "sequence_length": args.sequence_length,
        "save_model": not args.no_save,
        "verbose": not args.quiet,
    }
    
    # Train
    if len(tickers) == 1:
        results = train_single_ticker(tickers[0], **train_kwargs)
    else:
        results = train_multiple_tickers(tickers, **train_kwargs)
    
    print("\n✓ Training complete!")
    
    return results


if __name__ == "__main__":
    main()
