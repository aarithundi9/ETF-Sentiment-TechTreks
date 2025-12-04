"""Quick baseline comparison script."""
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/processed/QQQtotal.csv')
df['date'] = pd.to_datetime(df['date'])

# Create multi-horizon targets
df['target_5d'] = df['close'].pct_change(periods=5).shift(-5)
df['target_1m'] = df['close'].pct_change(periods=21).shift(-21)

# Time-based split (same as training)
df = df.dropna(subset=['target_5d', 'target_1m'])
split_idx = int(len(df) * 0.85)
test_df = df.iloc[split_idx:].copy()

print('='*60)
print('BASELINE COMPARISON - Test Set')
print('='*60)
print(f'Test period: {test_df["date"].min().date()} to {test_df["date"].max().date()}')
print(f'Test samples: {len(test_df)}')

# Actual directions
dir_5d = np.sign(test_df['target_5d'])
dir_1m = np.sign(test_df['target_1m'])

print(f'\n--- Actual Market Direction (Test Set) ---')
print(f'5D:  Up={sum(dir_5d>0)}, Down={sum(dir_5d<0)}, Flat={sum(dir_5d==0)}')
print(f'1M:  Up={sum(dir_1m>0)}, Down={sum(dir_1m<0)}, Flat={sum(dir_1m==0)}')

# Baseline 1: Always predict UP
always_up_5d = np.mean(dir_5d > 0) * 100
always_up_1m = np.mean(dir_1m > 0) * 100

# Baseline 2: Random (50%)
random_acc = 50.0

# Baseline 3: Momentum (predict same direction as last period)
mom_5d = np.sign(test_df['close'].pct_change(5))
mom_1m = np.sign(test_df['close'].pct_change(21))
momentum_5d = np.mean(mom_5d == dir_5d) * 100
momentum_1m = np.mean(mom_1m == dir_1m) * 100

print(f'\n--- Baseline Accuracies ---')
print(f'                        5-Day    1-Month')
print(f'Random (50/50):         {random_acc:.1f}%     {random_acc:.1f}%')
print(f'Always Predict UP:      {always_up_5d:.1f}%     {always_up_1m:.1f}%')
print(f'Momentum (same dir):    {momentum_5d:.1f}%     {momentum_1m:.1f}%')

print(f'\n--- YOUR MODEL ---')
print(f'Multi-Horizon LSTM:     59.5%     72.3%')

print(f'\n--- Model vs Best Baseline ---')
best_5d = max(always_up_5d, momentum_5d, random_acc)
best_1m = max(always_up_1m, momentum_1m, random_acc)
print(f'Best baseline:          {best_5d:.1f}%     {best_1m:.1f}%')
print(f'Model improvement:      {59.5-best_5d:+.1f}%    {72.3-best_1m:+.1f}%')

if 59.5 > best_5d:
    print(f'\n✓ 5D: Model BEATS best baseline by {59.5-best_5d:.1f}%')
else:
    print(f'\n✗ 5D: Model underperforms best baseline by {best_5d-59.5:.1f}%')

if 72.3 > best_1m:
    print(f'✓ 1M: Model BEATS best baseline by {72.3-best_1m:.1f}%')
else:
    print(f'✗ 1M: Model underperforms best baseline by {best_1m-72.3:.1f}%')
