"""
Quick comparison of sklearn classifiers vs baselines for ETF prediction.
Tests if simpler models can beat the "always predict up" baseline.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load QQQ data
df = pd.read_csv('data/processed/QQQtotal.csv')
df['date'] = pd.to_datetime(df['date'])

# Create targets
df['target_5d'] = df['close'].pct_change(periods=5).shift(-5)
df['target_1m'] = df['close'].pct_change(periods=21).shift(-21)
df['target_5d_dir'] = (df['target_5d'] > 0).astype(int)
df['target_1m_dir'] = (df['target_1m'] > 0).astype(int)

# Feature columns (use what's available)
feature_cols = [col for col in df.columns if col not in [
    'date', 'ticker', 'target', 'target_5d', 'target_1m', 
    'target_5d_dir', 'target_1m_dir', 'forward_return'
] and df[col].dtype in ['float64', 'int64']]

print(f"Using {len(feature_cols)} features")

# Drop NaN
df_clean = df.dropna(subset=feature_cols + ['target_5d_dir', 'target_1m_dir'])
print(f"Clean samples: {len(df_clean)}")

# Time-based split (80% train, 20% test)
split_idx = int(len(df_clean) * 0.80)
train_df = df_clean.iloc[:split_idx]
test_df = df_clean.iloc[split_idx:]

print(f"Train: {len(train_df)} samples ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
print(f"Test:  {len(test_df)} samples ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models to test
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=0.1),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
}

print("\n" + "="*70)
print("SKLEARN CLASSIFIERS vs BASELINES")
print("="*70)

for horizon, target_col in [('5-Day', 'target_5d_dir'), ('1-Month', 'target_1m_dir')]:
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    
    # Baselines
    always_up_acc = (y_test == 1).mean() * 100
    random_acc = 50.0
    
    print(f"\n{'─'*70}")
    print(f"{horizon} Horizon (Test: {len(y_test)} samples)")
    print(f"{'─'*70}")
    print(f"  Always UP baseline:     {always_up_acc:.1f}%")
    print(f"  Random baseline:        {random_acc:.1f}%")
    print()
    
    best_model = None
    best_acc = 0
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred) * 100
        
        beat_baseline = "✓ BEATS" if acc > always_up_acc else "✗"
        print(f"  {name:25} {acc:.1f}%  {beat_baseline}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = name
    
    print()
    if best_acc > always_up_acc:
        print(f"  🏆 Best: {best_model} ({best_acc:.1f}%) beats baseline by {best_acc - always_up_acc:.1f}%")
    else:
        print(f"  ⚠ No model beats 'Always UP' baseline ({always_up_acc:.1f}%)")

# Also try regression approach (predict return magnitude, then direction)
print("\n" + "="*70)
print("REGRESSION APPROACH (Predict return, check direction)")
print("="*70)

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

reg_models = {
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest Reg': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
}

for horizon, target_col, dir_col in [('5-Day', 'target_5d', 'target_5d_dir'), 
                                      ('1-Month', 'target_1m', 'target_1m_dir')]:
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    y_test_dir = test_df[dir_col]
    
    always_up_acc = (y_test_dir == 1).mean() * 100
    
    print(f"\n{horizon} Horizon:")
    print(f"  Always UP baseline:     {always_up_acc:.1f}%")
    
    for name, model in reg_models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_dir = (y_pred > 0).astype(int)
        acc = accuracy_score(y_test_dir, y_pred_dir) * 100
        
        beat_baseline = "✓ BEATS" if acc > always_up_acc else "✗"
        print(f"  {name:25} {acc:.1f}%  {beat_baseline}")
