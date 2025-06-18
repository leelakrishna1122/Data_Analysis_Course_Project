import pandas as pd
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
df = pd.read_csv("laptops_cleaned.csv")  # Make sure this file exists in DM_assignment/

# Split into 80% training and 20% testing
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Save the new datasets
train.to_csv("laptops_train.csv", index=False)
test.to_csv("laptops_test.csv", index=False)

print(f"Train set size: {len(train)} rows")
print(f"Test set size: {len(test)} rows")
