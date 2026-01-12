import pandas as pd
import os

# Create directory if not exists
os.makedirs("data/streaming_input", exist_ok=True)

# Read first 100 rows of source data
source_file = "data/yellow_tripdata_2023-01.csv"
if not os.path.exists(source_file):
    print(f"Error: {source_file} not found.")
    exit(1)

df = pd.read_csv(source_file, nrows=100)

# Write to streaming input without BOM
output_file = "data/streaming_input/stream_data.csv"
df.to_csv(output_file, index=False, encoding='utf-8')

print(f"Success! Written 100 rows to {output_file} with generic UTF-8 (no BOM).")
