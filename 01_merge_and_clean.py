import pandas as pd
import numpy as np
import os

DATA_PATH = "data"

files = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
]

dfs = []

for file in files:
    print(f"Loading {file}")

    df = pd.read_csv(
        os.path.join(DATA_PATH, file),
        encoding="ISO-8859-1",
        low_memory=False
    )

    df.columns = df.columns.str.strip()

    dfs.append(df)

full_df = pd.concat(
    dfs,
    ignore_index=True
)

print("\nMerged Shape:")
print(full_df.shape)

print("\nRemoving duplicates...")
full_df.drop_duplicates(inplace=True)

print("\nReplacing Inf values...")
full_df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

print("\nDropping NaNs...")
full_df.dropna(inplace=True)

print("\nFinal Shape:")
print(full_df.shape)

print("\nClass Distribution:")
print(full_df["Label"].value_counts())

full_df.to_csv(
    "outputs/cicids2017_clean.csv",
    index=False
)

print("\nSaved:")
print("../outputs/cicids2017_clean.csv")