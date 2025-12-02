"""View model weights and feature importance in detail."""

import pickle
import pandas as pd
import glob

# Load latest model
model_files = sorted(glob.glob('data/model_*.pkl'))
latest_model = model_files[-1]

with open(latest_model, 'rb') as f:
    model = pickle.load(f)

print(f"Model: {latest_model}")
print("=" * 80)

# Get coefficients and feature names
coef = model.model.coef_[0]  # Shape: (48,)
feature_names = model.feature_names
intercept = model.model.intercept_[0]

# Create DataFrame
weights_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coef,
    'abs_coefficient': abs(coef)
})

# Sort by absolute value
weights_df = weights_df.sort_values('abs_coefficient', ascending=False)

print(f"\n📊 FEATURE IMPORTANCE (sorted by absolute weight)")
print("=" * 80)
print(f"Intercept (bias): {intercept:.6f}\n")
print(weights_df.to_string(index=False))

print(f"\n\n💡 INTERPRETATION:")
print("=" * 80)
print("• Positive coefficient = feature increase → higher probability of price going UP")
print("• Negative coefficient = feature increase → higher probability of price going DOWN")
print("• Larger absolute value = stronger influence on prediction")
print(f"\nTop 3 most important features:")
for i, row in weights_df.head(3).iterrows():
    direction = "UP" if row['coefficient'] > 0 else "DOWN"
    print(f"  {row['feature']}: {row['coefficient']:.4f} → predicts {direction}")

# Save to CSV
output_file = 'model_weights.csv'
weights_df.to_csv(output_file, index=False)
print(f"\n✓ Weights saved to: {output_file}")
