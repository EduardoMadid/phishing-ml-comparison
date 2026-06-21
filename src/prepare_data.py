import pandas as pd
from sklearn.model_selection import train_test_split 

# Constants
DATA_PATH = 'data/phishing_site_urls.csv'
TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'

# Load the dataset
df = pd.read_csv(DATA_PATH)
print(f'Loaded {len(df)} rows from {DATA_PATH}')

# Drop duplicates
df = df.drop_duplicates()
print(f'After dropping duplicates, {len(df)} rows remain')

# Split into features and target
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["Label"], random_state=42)
print(f"Train: {len(train_df)} rows")
print(f"Test:  {len(test_df)} rows")

# Save to CSV
train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)
print(f"Saved {TRAIN_PATH} and {TEST_PATH}")