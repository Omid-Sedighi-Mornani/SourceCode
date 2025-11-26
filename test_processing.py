#!/usr/bin/env python3
"""
Test script to validate the processing pipeline changes
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# add projects root directory to the system path
sys.path.append(str(Path(".").resolve()))

# Load data
print("Loading data...")
reviews = pd.read_csv("data/reviews.csv")
business_covariates = pd.read_csv("data/business_covariates.csv")

print(f"Reviews shape: {reviews.shape}")
print(f"Business covariates shape: {business_covariates.shape}")

# Check columns after Age transformation
print("\nChecking Age column...")
if "Age" in business_covariates.columns:
    print("ERROR: Age already exists!")
    sys.exit(1)

# Test one-hot encoding with category
print("\nTesting one-hot encoding...")
test_df = pd.DataFrame({
    "density": [1, 2, 3],
    "Checkin": [0.5, 1.0, 1.5],
    "category": ["American", "Asian", "Cafes"],
    "chain": [0, 1, 0],
    "Price.Level": [1.0, 2.0, 1.0],
    "Restaurant.Size": [100.0, 200.0, 150.0],
    "Number.of.Seats": [50.0, 100.0, 75.0],
    "ZRI": [1500.0, 1600.0, 1550.0],
    "Age": [1000, 2000, 1500]
})

# Convert to category
test_df["category"] = test_df["category"].astype("category")

# One-hot encode
test_encoded = pd.get_dummies(test_df, columns=["category"], drop_first=False, dtype=int)

print(f"Columns after encoding: {test_encoded.columns.tolist()}")
print(f"Number of columns: {len(test_encoded.columns)}")

# Reorder columns
numeric_cols = ["density", "Checkin", "chain", "Price.Level", "Restaurant.Size",
                "Number.of.Seats", "ZRI", "Age"]
category_cols = [col for col in test_encoded.columns if col.startswith("category_")]

test_reordered = test_encoded[numeric_cols + sorted(category_cols)]
print(f"\nColumns after reordering: {test_reordered.columns.tolist()}")

# Remove column 8 (index 7)
col_to_remove = test_reordered.columns[7]
print(f"\nRemoving column at index 7 (R index 8): {col_to_remove}")
test_final = test_reordered.drop(columns=[col_to_remove])

print(f"Final columns: {test_final.columns.tolist()}")
print(f"Final shape: {test_final.shape}")

print("\n✓ Test completed successfully!")
