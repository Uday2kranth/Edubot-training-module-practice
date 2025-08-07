import pandas as pd

# Load the original dataset
df = pd.read_csv('fraud_0.1origbase.csv')

# Create a smaller sample (10,000 rows instead of full dataset)
sample_df = df.sample(n=10000, random_state=42)

# Save the sample
sample_df.to_csv('fraud_sample.csv', index=False)

print(f'Created sample with {len(sample_df)} rows')
print(f'Original size: {len(df)} rows')
print(f'Reduction: {(1 - len(sample_df)/len(df))*100:.1f}%')
