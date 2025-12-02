import pandas as pd

df = pd.read_csv('data/processed/modeling_dataset.csv')

print('Dataset Info:')
print(f'  Shape: {df.shape}')
print(f'  Tickers: {df["ticker"].unique()}')
print(f'  Date range: {df["date"].min()} to {df["date"].max()}')
print(f'\nTarget distribution:')
print(df['target'].value_counts())
print(f'\nSample features:')
print(df.columns.tolist()[:20])
