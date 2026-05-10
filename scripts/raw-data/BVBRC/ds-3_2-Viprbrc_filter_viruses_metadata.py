#!/usr/bin/env python3
"""
BV-BRC Metadata Filter

Filters and cleans the BV-BRC metadata by keeping only the latest version
of each virus genome. Removes older versions to avoid duplicates.

Usage:
    python raw-data/BVBRC/ds-3_2-Viprbrc_filter_viruses_metadata.py

Requirements:
    - pandas library
    - all_hosts_viruses_metadata_noSeq.csv (from ds-3_1)

Input:
    - raw-data/BVBRC/all_hosts_viruses_metadata_noSeq.csv

Output:
    - raw-data/BVBRC/all_hosts_viruses_metadata_noSeq_filtered.csv

Notes:
    - Splits Virus_ID into Vir_ID and Version
    - Keeps only the row with maximum Version for each Vir_ID
"""

import pandas as pd
import os
import time
from config import RAW_DATA_BVBRC_PATH

# Define file paths
input_folder = RAW_DATA_BVBRC_PATH
input_file = "all_hosts_viruses_metadata_noSeq.csv"
output_file = "all_hosts_viruses_metadata_noSeq_filtered.csv"

input_path = os.path.join(input_folder, input_file)
output_path = os.path.join(input_folder, output_file)

# Record the start time
start_time = time.time()

# Read the CSV file (comma-separated)
df = pd.read_csv(input_path)

# Split Virus_ID into Vir_ID and Version
split_cols = df['Virus_ID'].astype(str).str.split('.', n=1, expand=True)
df['Vir_ID'] = split_cols[0]
df['Version'] = split_cols[1].astype(int)

# For each unique Vir_ID, keep row(s) with the maximum Version
max_version = df.groupby('Vir_ID')['Version'].transform('max')
filtered_df = df[df['Version'] == max_version]

# Save the filtered dataframe
filtered_df.to_csv(output_path, index=False)

# Record the end time
end_time = time.time()

# Calculate the elapsed time in minutes
elapsed_time = (end_time - start_time) / 60

print(f"Filtered data saved to: {output_path}")
print(f"Running time: {elapsed_time:.2f} minutes")
