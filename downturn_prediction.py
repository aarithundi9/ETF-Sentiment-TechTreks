"""
Alternative approach: Predict DOWNTURNS (when market will drop).
This is more useful for risk management - even if you can't beat "always up",
identifying downturns has value for hedging/avoiding losses.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load QQQ data
df = pd.read_csv('data/processed/QQQtotal.csv')
df['date'] = pd.to_datetime(df['date'])

# Create targets - predict SIGNIFICANT drops (> 2% for 5d, > 5% for 1m)
df['return_5d'] = df['close'].pct_change(periods=5).shift(-5)
df['return_1m'] = df['close'].pct_change(periods=21).shift(-21)

# Binary targets: 1 = significant drop, 0 = otherwise
df['drop_5d'] = (df['return_5d'] < -0.02).astype(int)  # 5d drop > 2%
df['drop_1m'] = (df['return_1m'] < -0.05).astype(int)  # 1m drop > 5%

# Feature columns
feature_cols = [col for col in df.columns if col not in [
    'date', 'ticker', 'target', 'return_5d', 'return_1m', 
    'drop_5d', 'drop_1m', 'forward_return', 'target_5d', 'target_1m'
] and df[col].dtype in ['float64', 'int64']]

# Drop NaN
df_clean = df.dropna(subset=feature_cols + ['drop_5d', 'drop_1m'])

# Time-based split
split_idx = int(len(df_clean) * 0.80)
train_df = df_clean.iloc[:split_idx]
test_df = df_clean.iloc[split_idx:]

print("="*70)
print("DOWNTURN PREDICTION MODEL")
print("="*70)
print(f"Train: {len(train_df)} samples ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
print(f"Test:  {len(test_df)} samples ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Class-weighted models (handle imbalanced data)
models = {
    'Logistic (balanced)': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Random Forest (balanced)': RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
}

for horizon, target_col, threshold in [('5-Day (>2% drop)', 'drop_5d', 0.02), 
                                        ('1-Month (>5% drop)', 'drop_1m', 0.05)]:
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    
    print(f"\n{'─'*70}")
    print(f"{horizon}")
    print(f"{'─'*70}")
    print(f"  Actual drops in test set: {y_test.sum()}/{len(y_test)} ({y_test.mean()*100:.1f}%)")
    print()
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"  {name}:")
        print(f"    Accuracy:  {acc*100:.1f}%")
        print(f"    Precision: {prec*100:.1f}% (of predicted drops, how many were real)")
        print(f"    Recall:    {rec*100:.1f}% (of real drops, how many did we catch)")
        print(f"    F1 Score:  {f1*100:.1f}%")
        print(f"    Confusion: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
        print()

# Value proposition
print("="*70)
print("WHY THIS MATTERS FOR PRESENTATION")
print("="*70)
print("""
Even if we can't beat 'always up' for direction prediction, 
DOWNTURN prediction has real value:

1. RISK MANAGEMENT: Catch even 30-40% of drops → reduce portfolio volatility
2. HEDGING SIGNALS: Use predictions to adjust position sizes
3. ASYMMETRIC VALUE: Avoiding a 5% loss > missing a 5% gain

The model doesn't need to be perfect - it needs to be USEFUL.
""")

# Feature importance
print("\n" + "="*70)
print("TOP FEATURES FOR PREDICTING DROPS")
print("="*70)

# Train RF on 1-month drops
rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
rf.fit(X_train_scaled, train_df['drop_1m'])

importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 features for predicting 1-month drops:")
for i, row in importances.head(10).iterrows():
    print(f"  {row['feature']:25} {row['importance']:.4f}")
