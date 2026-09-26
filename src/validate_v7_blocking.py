import pandas as pd
import re
import unicodedata
import random
from collections import defaultdict


BASE = r"C:\Users\Lenovo\amazon-ml-challenge-2026-data"
TRAIN = BASE + r"\student_resource\dataset\train"

SAMPLE_SIZE = 5000
RANDOM_SEED = 42


def norm_text(value):
    value = unicodedata.normalize("NFKD", str(value))
    value = value.encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", value)).strip()


def name_tokens(value):
    text = norm_text(value)

    suffixes = {
        "inc", "incorporated", "corp", "corporation",
        "company", "co", "ltd", "limited", "llc", "llp"
    }

    tokens = text.split()

    while tokens and tokens[-1] in suffixes:
        tokens.pop()

    return tokens


def address_info(value):
    text = norm_text(value)
    tokens = text.split()

    numbers = set(re.findall(r"\b\d+[a-z]?\b", text))

    common = {
        "street", "road", "avenue", "boulevard", "drive",
        "court", "lane", "place", "circle", "way",
        "trail", "parkway", "highway", "suite", "floor",
        "apartment", "building", "room", "number", "near",
        "opposite", "behind", "block", "sector", "phase",
        "plot", "flat", "door", "fl", "no", "unit",
        "north", "south", "east", "west", "india", "usa",
        "us", "france", "state", "district", "city",
        "nagar", "colony", "bazaar", "marg", "gali",
        "rd", "st", "ave", "blvd", "dr", "ct", "ln",
        "hwy", "apt", "ste"
    }

    significant = [
        t for t in tokens
        if len(t) >= 3
        and not t.isdigit()
        and t not in common
    ]

    city = tokens[-1] if tokens else ""

    return text, numbers, significant, city


def make_keys(name, address, country):
    nt = name_tokens(name)
    addr, nums, sig, city = address_info(address)

    keys = set()

    # Existing-style compact name
    compact = "".join(nt)

    if len(compact) >= 4:
        keys.add(("compact_name", country, compact))

    # NEW composite keys
    if nt and nums:
        for num in nums:
            keys.add(("name_num", country, nt[0], num))

    if nums and sig:
        for num in nums:
            for street in sig[:3]:
                keys.add(("num_street", country, num, street))

    if nums and city:
        for num in nums:
            keys.add(("num_city", country, num, city))

    return keys


print("Loading training data...")

s1 = pd.read_csv(
    TRAIN + r"\train_source1.tsv",
    sep="\t",
    usecols=["entity_id", "business_name", "business_address", "country"],
    dtype=str
)

s2 = pd.read_csv(
    TRAIN + r"\train_source2.tsv",
    sep="\t",
    usecols=["entity_id", "business_name", "business_address", "country"],
    dtype=str
)

s3 = pd.read_csv(
    TRAIN + r"\train_source3.tsv",
    sep="\t",
    usecols=["entity_id", "business_name", "business_address", "country"],
    dtype=str
)

gt = pd.read_csv(
    TRAIN + r"\train_ground_truth.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print(f"S1: {len(s1):,}")
print(f"S2: {len(s2):,}")
print(f"S3: {len(s3):,}")

# ------------------------------------------------------------
# Random 5K validation sample
# ------------------------------------------------------------

sample = s1.sample(
    n=SAMPLE_SIZE,
    random_state=RANDOM_SEED
)

sample_ids = set(sample["entity_id"])

gt_sample = gt[
    gt["source1_entity_id"].isin(sample_ids)
]

truth = {}

for row in gt_sample.itertuples(index=False):
    ids = {
        x.strip()
        for x in row.matched_entity_ids.split(",")
        if x.strip()
    }
    truth[row.source1_entity_id] = ids


# ------------------------------------------------------------
# Build indexes
# ------------------------------------------------------------

print("Building blocking indexes...")

compact_index = defaultdict(set)
name_num_index = defaultdict(set)
num_street_index = defaultdict(set)
num_city_index = defaultdict(set)

for df in [s2, s3]:

    for row in df.itertuples(index=False):

        keys = make_keys(
            row.business_name,
            row.business_address,
            row.country
        )

        for key in keys:

            if key[0] == "compact_name":
                compact_index[key].add(row.entity_id)

            elif key[0] == "name_num":
                name_num_index[key].add(row.entity_id)

            elif key[0] == "num_street":
                num_street_index[key].add(row.entity_id)

            elif key[0] == "num_city":
                num_city_index[key].add(row.entity_id)


print("Testing candidate recall...")

old_recovered = 0
new_recovered = 0

recovered_by_name_num = 0
recovered_by_num_street = 0
recovered_by_num_city = 0

total_truth = 0

for row in sample.itertuples(index=False):

    sid = row.entity_id
    actual = truth.get(sid, set())

    if not actual:
        continue

    total_truth += len(actual)

    keys = make_keys(
        row.business_name,
        row.business_address,
        row.country
    )

    old_candidates = set()
    new_candidates = set()

    for key in keys:

        if key[0] == "compact_name":
            old_candidates.update(
                compact_index.get(key, ())
            )

        elif key[0] == "name_num":
            new_candidates.update(
                name_num_index.get(key, ())
            )

        elif key[0] == "num_street":
            new_candidates.update(
                num_street_index.get(key, ())
            )

        elif key[0] == "num_city":
            new_candidates.update(
                num_city_index.get(key, ())
            )

    old_hit = actual & old_candidates

    new_hit = actual & new_candidates

    old_recovered += len(old_hit)

    # Only count genuinely new recovery
    new_only = new_hit - old_candidates

    new_recovered += len(new_only)

    if new_only:

        # Determine which keys recovered them
        for key in keys:

            if key[0] == "name_num":
                if new_only & name_num_index.get(key, set()):
                    recovered_by_name_num += 1

            elif key[0] == "num_street":
                if new_only & num_street_index.get(key, set()):
                    recovered_by_num_street += 1

            elif key[0] == "num_city":
                if new_only & num_city_index.get(key, set()):
                    recovered_by_num_city += 1


print()
print("=" * 60)
print("V7 BLOCKING VALIDATION")
print("=" * 60)

print(f"Sample S1 records:        {SAMPLE_SIZE:,}")
print(f"Ground-truth matches:     {total_truth:,}")
print()

old_recall = (
    old_recovered / total_truth
    if total_truth else 0
)

new_recall = (
    (old_recovered + new_recovered) / total_truth
    if total_truth else 0
)

print(f"Existing name recall:     {old_recall:.4%}")
print(f"V7 blocking recall:       {new_recall:.4%}")
print()

print("NEW TRUE MATCHES RECOVERED")
print(f"name + number:            {recovered_by_name_num:,}")
print(f"number + street:          {recovered_by_num_street:,}")
print(f"number + city:            {recovered_by_num_city:,}")
print()

print(f"Additional recovery:      {new_recovered:,}")

print("=" * 60)