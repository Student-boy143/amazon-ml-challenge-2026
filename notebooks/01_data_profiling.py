import pandas as pd

DATA_DIR = r"C:\Users\Lenovo\amazon-ml-challenge-2026-data\student_resource\dataset\train"

files = {
    "source1": f"{DATA_DIR}\\train_source1.tsv",
    "source2": f"{DATA_DIR}\\train_source2.tsv",
    "source3": f"{DATA_DIR}\\train_source3.tsv",
}

for name, path in files.items():
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=["country"]
    )

    print(f"\n{name.upper()}")
    print(f"Total rows: {len(df):,}")
    print(df["country"].value_counts(dropna=False).head(20))

for name, path in files.items():
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=["business_name"]
    )

    total = len(df)
    unique = df["business_name"].nunique()
    duplicates = total - unique

    print(f"\n{name.upper()} NAME STATISTICS")
    print(f"Total records: {total:,}")
    print(f"Unique names: {unique:,}")
    print(f"Duplicate-name records: {duplicates:,}")
    print(f"Unique-name ratio: {unique / total:.2%}")

import pandas as pd
import os

BASE = r"C:\Users\Lenovo\amazon-ml-challenge-2026-data\student_resource"

files = {
    "test_source1": "test_source1.tsv",
    "test_source2": "test_source2.tsv",
    "test_source3": "test_source3.tsv",
}

for name, filename in files.items():
    path = os.path.join(BASE, "dataset", "test", filename)

    df = pd.read_csv(path, sep="\t", usecols=["entity_id"])

    print(f"{name.upper()}")
    print(f"Rows: {len(df):,}")
    print()