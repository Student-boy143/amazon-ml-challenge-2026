import os
import re
import pandas as pd


BASE = r"C:\Users\Lenovo\amazon-ml-challenge-2026-data\student_resource"
TEST_DIR = os.path.join(BASE, "dataset", "test")
OUTPUT_DIR = r"C:\Users\Lenovo\amazon-ml-challenge-2026\submissions\name_address_v1"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()

    # Normalize common separators/punctuation
    value = re.sub(r"[^a-z0-9]+", " ", value)

    # Common business/legal suffixes
    value = re.sub(
        r"\b(incorporated|inc|corporation|corp|limited|ltd|llc|llp|"
        r"company|co)\b",
        " ",
        value
    )

    return " ".join(value.split())


print("Loading test data...")

s1 = pd.read_csv(
    os.path.join(TEST_DIR, "test_source1.tsv"),
    sep="\t",
    usecols=["entity_id", "business_name", "business_address", "country"]
)

s2 = pd.read_csv(
    os.path.join(TEST_DIR, "test_source2.tsv"),
    sep="\t",
    usecols=["entity_id", "business_name", "business_address", "country"]
)

s3 = pd.read_csv(
    os.path.join(TEST_DIR, "test_source3.tsv"),
    sep="\t",
    usecols=["entity_id", "business_name", "business_address", "country"]
)

print(f"Source 1: {len(s1):,}")
print(f"Source 2: {len(s2):,}")
print(f"Source 3: {len(s3):,}")


print("Normalizing name + address...")

for df in (s1, s2, s3):
    df["name_norm"] = df["business_name"].map(normalize)
    df["address_norm"] = df["business_address"].map(normalize)


print("Building exact name + address index...")

index = {}

for df in (s2, s3):
    for row in df.itertuples(index=False):
        if not row.name_norm or not row.address_norm:
            continue

        key = (
            row.country,
            row.name_norm,
            row.address_norm
        )

        index.setdefault(key, []).append(row.entity_id)


print("Generating matches...")

matching_path = os.path.join(
    OUTPUT_DIR,
    "matching_results.tsv"
)

candidate_path = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs.tsv"
)

matched_entities = 0
total_matches = 0

with open(matching_path, "w", encoding="utf-8", newline="") as mf, \
     open(candidate_path, "w", encoding="utf-8", newline="") as cf:

    mf.write("source1_entity_id\tmatched_entity_ids\n")
    cf.write("source1_entity_id\tcandidate_entity_ids\n")

    for row in s1.itertuples(index=False):

        key = (
            row.country,
            row.name_norm,
            row.address_norm
        )

        matches = list(dict.fromkeys(index.get(key, [])))

        if matches:
            matched_entities += 1
            total_matches += len(matches)

        match_string = ",".join(matches)

        mf.write(
            f"{row.entity_id}\t{match_string}\n"
        )

        cf.write(
            f"{row.entity_id}\t{match_string}\n"
        )


print()
print("=== NAME + ADDRESS BASELINE ===")
print(f"Source 1 entities:        {len(s1):,}")
print(f"Entities with matches:    {matched_entities:,}")
print(f"Total matched IDs:        {total_matches:,}")
print(f"Output directory:         {OUTPUT_DIR}")