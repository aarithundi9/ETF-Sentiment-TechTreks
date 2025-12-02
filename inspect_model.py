"""Inspect the contents of a saved model file."""

import pickle
import glob
from pathlib import Path

# Find latest model
model_files = sorted(glob.glob('data/model_*.pkl'))
if not model_files:
    print("No model files found!")
    exit()

latest_model = model_files[-1]
print(f"Inspecting: {latest_model}")
print("=" * 80)

# Load model
with open(latest_model, 'rb') as f:
    model = pickle.load(f)

print(f"\n📦 Model Object Type: {type(model)}")
print(f"   Class: {model.__class__.__name__}")

# Show attributes
print(f"\n📋 Model Attributes:")
for attr in dir(model):
    if not attr.startswith('_'):
        value = getattr(model, attr)
        if not callable(value):
            print(f"   {attr}: {type(value).__name__}")

# Show the actual sklearn model
print(f"\n🤖 Scikit-Learn Model:")
print(f"   Type: {type(model.model)}")
print(f"   Algorithm: {model.model.__class__.__name__}")

# Show model parameters
if hasattr(model.model, 'get_params'):
    print(f"\n⚙️  Model Parameters:")
    params = model.model.get_params()
    for key, value in list(params.items())[:10]:  # First 10 params
        print(f"   {key}: {value}")

# Show coefficients (weights)
if hasattr(model.model, 'coef_'):
    coef = model.model.coef_
    print(f"\n⚖️  Model Weights (Coefficients):")
    print(f"   Shape: {coef.shape}")
    print(f"   Number of features: {coef.shape[1]}")
    print(f"   First 5 weights: {coef[0][:5]}")
    print(f"   Last 5 weights: {coef[0][-5:]}")

# Show intercept
if hasattr(model.model, 'intercept_'):
    print(f"\n📍 Intercept (bias): {model.model.intercept_}")

# Show scaler info
if hasattr(model, 'scaler') and model.scaler:
    print(f"\n📏 Feature Scaler:")
    print(f"   Type: {type(model.scaler).__name__}")
    if hasattr(model.scaler, 'mean_'):
        print(f"   Number of features scaled: {len(model.scaler.mean_)}")
        print(f"   Feature means (first 5): {model.scaler.mean_[:5]}")
        print(f"   Feature stds (first 5): {model.scaler.scale_[:5]}")

# Show feature names if available
if hasattr(model, 'feature_names') and model.feature_names:
    print(f"\n🏷️  Feature Names ({len(model.feature_names)} total):")
    print(f"   First 10: {model.feature_names[:10]}")
    print(f"   Last 10: {model.feature_names[-10:]}")

# Show metadata
print(f"\n📊 Model Metadata:")
if hasattr(model, 'model_type'):
    print(f"   Model Type: {model.model_type}")
if hasattr(model, 'trained_on'):
    print(f"   Trained On: {model.trained_on}")

# File size
file_size = Path(latest_model).stat().st_size
print(f"\n💾 File Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")

print("\n" + "=" * 80)
print("✓ Inspection complete!")
