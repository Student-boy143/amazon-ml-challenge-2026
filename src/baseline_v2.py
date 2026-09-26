import os
import re
import pandas as pd


BASE = r"C:\Users\Lenovo\amazon-ml-challenge-2026-data\student_resource"
TEST_DIR = os.path.join(BASE, "dataset", "test")
OUTPUT_DIR = r"C:\Users\Lenovo\amazon-ml-challenge-2026\submissions\name_address_v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_name(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)

    # Remove common legal suffixes from business names.
    value = re.sub(
        r"\b(incorporated|inc|corporation|corp|limited|ltd|llc|llp|company|co)\b",
        " ",
        value,
    )

    return " ".join(value.split())


def normalize_address(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)

    return " ".join(value.split())


def address_similarity(a, b):
    a_tokens = set(a.split())
    b_tokens = set(b.split())

    if not a_tokens or not b_tokens:
        return 0.0

    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)

    return intersection / union


print("Loading test data...")

s1 = pd.read_csv(
    os.path.join(TEST_DIR, "test_source1.tsv"),
    sep="\t",
    usecols=["entity_id", "business_name", "business_address", "country"],
)

s2 = pd.read_csv(
    os.path.join(TEST_DIR, "test_source2.tsv"),
    sep="\t",
    usecols=["entity_id", "business_name", "business_address", "country"],
)

s3 = pd.read_csv(
    os.path.join(TEST_DIR, "test_source3.tsv"),
    sep="\t",
    usecols=["entity_id", "business_name", "business_address", "country"],
)

print(f"Source 1: {len(s1):,}")
print(f"Source 2: {len(s2):,}")
print(f"Source 3: {len(s3):,}")


print("Normalizing...")

for df in (s1, s2, s3):
    df["name_norm"] = df["business_name"].map(normalize_name)
    df["address_norm"] = df["business_address"].map(normalize_address)


print("Building name index...")

index = {}

for df in (s2, s3):
    for row in df.itertuples(index=False):

        if not row.name_norm:
            continue

        key = (row.country, row.name_norm)

        index.setdefault(key, []).append(
            (row.entity_id, row.address_norm)
        )


print("Generating matches...")

matching_path = os.path.join(
    OUTPUT_DIR,
    "matching_results.tsv",
)

candidate_path = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs.tsv",
)

matched_entities = 0
total_matches = 0

# Conservative threshold.
THRESHOLD = 0.70


with open(matching_path, "w", encoding="utf-8", newline="") as mf, \
     open(candidate_path, "w", encoding="utf-8", newline="") as cf:

    mf.write("source1_entity_id\tmatched_entity_ids\n")
    cf.write("source1_entity_id\tcandidate_entity_ids\n")

    for i, row in enumerate(s1.itertuples(index=False), start=1):

        key = (row.country, row.name_norm)

        candidates = index.get(key, [])

        matches = []

        for candidate_id, candidate_address in candidates:

            score = address_similarity(
                row.address_norm,
                candidate_address,
            )

            if score >= THRESHOLD:
                matches.append(candidate_id)

        matches = list(dict.fromkeys(matches))

        if matches:
            matched_entities += 1
            total_matches += len(matches)

        result = ",".join(matches)

        mf.write(
            f"{row.entity_id}\t{result}\n"
        )

        cf.write(
            f"{row.entity_id}\t{result}\n"
        )

        if i % 250000 == 0:
            print(f"Processed {i:,} / {len(s1):,}")


print()
print("=== V2 RESULT ===")
print(f"Threshold:              {THRESHOLD}")
print(f"Entities with matches:  {matched_entities:,}")
print(f"Total matched IDs:      {total_matches:,}")
print(f"Output:                 {OUTPUT_DIR}")