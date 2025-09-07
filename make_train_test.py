import json
import random

file_path = "scifi_full.jsonl"
data = []

# Read JSONL file
with open(file_path, 'r') as file:
    for line in file:
        data.append(json.loads(line))

# Shuffle and split
random.shuffle(data)
split_ratio = 0.8
split_index = int(len(data) * split_ratio)
train_data = data[:split_index]
test_data = data[split_index:]

# Write to JSON files (as lists)
with open("scifi_train.json", 'w') as train_file:
    json.dump(train_data, train_file, indent=2)

with open("scifi_test.json", 'w') as test_file:
    json.dump(test_data, test_file, indent=2)

