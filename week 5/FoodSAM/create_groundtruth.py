import csv

metadata1 = "nutrition5k_dataset/metadata/dish_metadata_cafe1.csv"
metadata2 = "nutrition5k_dataset/metadata/dish_metadata_cafe2.csv"

# Step 1: Load sample dish IDs
with open("nutrition5k_dataset/dish_ids/splits/sample_dish_ids.txt") as f:
    sample_ids = set(line.strip() for line in f)

# Step 2: Extract only first 6 fields for matching dish IDs
groundtruth_rows = []

with open(metadata1, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] in sample_ids:
            trimmed_row = row[:6]  # Keep only first 6 fields
            groundtruth_rows.append(trimmed_row)

with open(metadata2, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] in sample_ids:
            trimmed_row = row[:6]  # Keep only first 6 fields
            groundtruth_rows.append(trimmed_row)

# Step 3: Save to new groundtruth CSV
with open("groundtruth_sample.csv", "w", newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(["dish_id", "calories", "mass", "fat", "carb", "protein"])  # header
    writer.writerows(groundtruth_rows)
