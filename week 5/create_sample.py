import pandas as pd

# Read only the first column
df1 = pd.read_csv("dish_metadata_cafe1.csv", usecols=[0], names=["dish_id"], skiprows=1)
df2 = pd.read_csv("dish_metadata_cafe2.csv", usecols=[0], names=["dish_id"], skiprows=1)

# Combine and sample
df = pd.concat([df1, df2], ignore_index=True)
sample_df = df.sample(n=50, random_state=42)

# Save the dish IDs
sample_df.to_csv("sample_dish_ids.txt", index=False, header=False)

#next, download samples