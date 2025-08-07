import pandas as pd

# Load the original dataset
df = pd.read_csv('fraud_0.1origbase.csv')

# Create a larger sample (half of original dataset - around 300K rows)
sample_size = len(df) // 2  # Half of the original dataset
sample_df = df.sample(n=sample_size, random_state=42)

# Save the sample
sample_df.to_csv('fraud_sample.csv', index=False)

print(f'Created sample with {len(sample_df):,} rows')
print(f'Original size: {len(df):,} rows')
print(f'Sample size: {sample_size:,} rows (50% of original)')
print(f'File size reduction: {(1 - len(sample_df)/len(df))*100:.1f}%')
