import os
import re
import pandas as pd


BASE = r"C:\Users\Lenovo\amazon-ml-challenge-2026-data\student_resource"

TRAIN = os.path.join(BASE, "dataset", "train")


def normalize_name(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


print("Loading training data...")

s1 = pd.read_csv(
    os.path.join(TRAIN, "train_source1.tsv"),
    sep="\t",
    usecols=["entity_id", "business_name", "country"]
)

s2 = pd.read_csv(
    os.path.join(TRAIN, "train_source2.tsv"),
    sep="\t",
    usecols=["entity_id", "business_name", "country"]
)

s3 = pd.read_csv(
    os.path.join(TRAIN, "train_source3.tsv"),
    sep="\t",
    usecols=["entity_id", "business_name", "country"]
)

gt = pd.read_csv(
    os.path.join(TRAIN, "train_ground_truth.tsv"),
    sep="\t"
)

for df in (s1, s2, s3):
    df["name_norm"] = df["business_name"].map(normalize_name)


print("Building indexes...")

indexes = {}

for df in (s2, s3):
    for row in df.itertuples(index=False):
        key = (row.country, row.name_norm)

        if row.name_norm:
            indexes.setdefault(key, set()).add(row.entity_id)


print("Evaluating...")

total_true = 0
found_true = 0
entities_with_true_match = 0
entities_with_found_match = 0

for row in gt.itertuples(index=False):

    true_ids = set(
        x.strip()
        for x in str(row.matched_entity_ids).split(",")
        if x.strip()
    )

    key_row = s1.loc[s1["entity_id"] == row.source1_entity_id].iloc[0]

    key = (key_row.country, key_row.name_norm)

    predicted_ids = indexes.get(key, set())

    total_true += len(true_ids)

    overlap = true_ids & predicted_ids

    found_true += len(overlap)

    if true_ids:
        entities_with_true_match += 1

    if overlap:
        entities_with_found_match += 1


print()
print("=== BASELINE EVALUATION ===")
print(f"True matched IDs:              {total_true:,}")
print(f"True IDs recovered:             {found_true:,}")
print(f"Match-level recall:             {found_true / total_true:.2%}")
print(f"Entities with true matches:     {entities_with_true_match:,}")
print(f"Entities with >=1 match found: {entities_with_found_match:,}")
print(
    f"Entity-level recall:            "
    f"{entities_with_found_match / entities_with_true_match:.2%}"
)