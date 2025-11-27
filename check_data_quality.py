#!/usr/bin/env python3
"""
Überprüft die Datenqualität für Stan-Modelle
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pickle
import numpy as np
from helpers import ModelData

def check_data_quality(data_path):
    """
    Überprüft ob die Daten für Stan geeignet sind
    """
    print("="*60)
    print("DATENQUALITÄTS-CHECK FÜR STAN")
    print("="*60)

    # Load data
    model_data = ModelData.from_pickle(data_path)

    print("\n1. GRUNDLEGENDE STATISTIKEN")
    print("-"*60)
    print(f"N_total: {model_data.n_total}")
    print(f"N_train: {model_data.n_train}")
    print(f"N_test: {model_data.n_total - model_data.n_train}")
    print(f"N_obs (total ratings): {model_data.n_obs}")
    print(f"nCovs: {model_data.n_covs}")

    print("\n2. RATINGS (1-5)")
    print("-"*60)
    ratings_arr = np.array(model_data.ratings)
    unique_ratings = np.unique(ratings_arr)
    print(f"Unique ratings: {unique_ratings}")
    print(f"Rating counts:")
    for r in range(1, 6):
        count = np.sum(ratings_arr == r)
        pct = 100 * count / len(ratings_arr)
        print(f"  Rating {r}: {count} ({pct:.1f}%)")

    # Check for invalid ratings
    invalid_ratings = np.sum((ratings_arr < 1) | (ratings_arr > 5))
    if invalid_ratings > 0:
        print(f"⚠️  WARNING: {invalid_ratings} invalid ratings found!")
    else:
        print("✓ All ratings in valid range [1,5]")

    print("\n3. DAYS (Zeit seit erster Review)")
    print("-"*60)
    days_arr = np.array(model_data.days)
    print(f"Min days: {np.min(days_arr):.2f}")
    print(f"Max days: {np.max(days_arr):.2f}")
    print(f"Mean days: {np.mean(days_arr):.2f}")
    print(f"Median days: {np.median(days_arr):.2f}")

    # Check for negative or NaN
    if np.any(days_arr < 0):
        print("⚠️  WARNING: Negative days found!")
    if np.any(np.isnan(days_arr)):
        print("⚠️  WARNING: NaN values in days!")
    else:
        print("✓ All days >= 0 and non-NaN")

    print("\n4. SENTIMENT")
    print("-"*60)
    sentiment_arr = np.array(model_data.sentiment)
    print(f"Min sentiment: {np.min(sentiment_arr):.4f}")
    print(f"Max sentiment: {np.max(sentiment_arr):.4f}")
    print(f"Mean sentiment: {np.mean(sentiment_arr):.4f}")
    print(f"Std sentiment: {np.std(sentiment_arr):.4f}")

    # Check for extreme values
    extreme_sentiment = np.sum(np.abs(sentiment_arr) > 10)
    if extreme_sentiment > 0:
        print(f"⚠️  WARNING: {extreme_sentiment} extreme sentiment values (|x| > 10)")
    if np.any(np.isnan(sentiment_arr)):
        print("⚠️  WARNING: NaN values in sentiment!")
    else:
        print("✓ No NaN in sentiment")

    print("\n5. TIME (Anzahl Reviews pro Restaurant)")
    print("-"*60)
    time_arr = np.array(model_data.time)
    print(f"Min reviews per restaurant: {np.min(time_arr)}")
    print(f"Max reviews per restaurant: {np.max(time_arr)}")
    print(f"Mean reviews per restaurant: {np.mean(time_arr):.1f}")
    print(f"Median reviews per restaurant: {np.median(time_arr):.1f}")

    # Check if sum matches
    total_reviews = np.sum(time_arr)
    if total_reviews != model_data.n_obs:
        print(f"⚠️  ERROR: sum(Time) = {total_reviews} != N_obs = {model_data.n_obs}")
    else:
        print(f"✓ sum(Time) = {total_reviews} matches N_obs")

    # Check for restaurants with very few reviews
    few_reviews = np.sum(time_arr < 3)
    if few_reviews > 0:
        print(f"⚠️  INFO: {few_reviews} restaurants with < 3 reviews (may cause problems)")

    print("\n6. CLOSED STATUS")
    print("-"*60)
    closed_arr = np.array(model_data.closed)
    n_closed = np.sum(closed_arr)
    n_open = len(closed_arr) - n_closed
    print(f"Closed: {n_closed} ({100*n_closed/len(closed_arr):.1f}%)")
    print(f"Open: {n_open} ({100*n_open/len(closed_arr):.1f}%)")

    # Check for class imbalance
    if n_closed < 0.1 * len(closed_arr) or n_closed > 0.9 * len(closed_arr):
        print("⚠️  WARNING: Strong class imbalance! (< 10% or > 90%)")
    else:
        print("✓ Reasonable class balance")

    print("\n7. COVARIATES (Q, R, X_test)")
    print("-"*60)
    print(f"Q shape: {model_data.Q.shape} (should be {model_data.n_train} x {model_data.n_covs})")
    print(f"R shape: {model_data.R.shape} (should be {model_data.n_covs} x {model_data.n_covs})")
    print(f"X_test shape: {model_data.X_test.shape} (should be {model_data.n_total - model_data.n_train} x {model_data.n_covs})")

    # Check Q
    if np.any(np.isnan(model_data.Q)):
        print("⚠️  WARNING: NaN in Q!")
    if np.any(np.isinf(model_data.Q)):
        print("⚠️  WARNING: Inf in Q!")

    print(f"\nQ statistics:")
    print(f"  Mean: {np.mean(model_data.Q):.4f}")
    print(f"  Std: {np.std(model_data.Q):.4f}")
    print(f"  Min: {np.min(model_data.Q):.4f}")
    print(f"  Max: {np.max(model_data.Q):.4f}")

    # Check if Q is roughly standardized (from QR decomposition)
    if np.abs(np.mean(model_data.Q)) > 1:
        print("⚠️  INFO: Q is not centered (expected after QR decomposition)")

    # Check R
    if np.any(np.isnan(model_data.R)):
        print("⚠️  WARNING: NaN in R!")
    if np.any(np.isinf(model_data.R)):
        print("⚠️  WARNING: Inf in R!")

    # Check if R is upper triangular (from QR)
    is_upper_triangular = np.allclose(model_data.R, np.triu(model_data.R))
    if is_upper_triangular:
        print("✓ R is upper triangular (correct QR decomposition)")
    else:
        print("⚠️  WARNING: R is not upper triangular!")

    # Check for rank deficiency
    r_rank = np.linalg.matrix_rank(model_data.R)
    if r_rank < model_data.n_covs:
        print(f"⚠️  WARNING: R is rank deficient! Rank = {r_rank} < {model_data.n_covs}")
    else:
        print(f"✓ R has full rank ({r_rank})")

    # Check condition number
    cond_number = np.linalg.cond(model_data.R)
    print(f"\nR condition number: {cond_number:.2e}")
    if cond_number > 1e10:
        print("⚠️  WARNING: R is very ill-conditioned! (cond > 1e10)")
        print("   This can cause numerical instability in Stan!")
    elif cond_number > 1e6:
        print("⚠️  INFO: R is somewhat ill-conditioned (cond > 1e6)")
    else:
        print("✓ R is well-conditioned")

    print("\n8. POTENTIELLE STAN-PROBLEME")
    print("-"*60)

    issues = []

    # Check for extreme covariate values
    if np.max(np.abs(model_data.Q)) > 100:
        issues.append("Extreme values in Q (|x| > 100)")

    # Check sentiment scale
    if np.std(model_data.sentiment) > 10:
        issues.append("Very large sentiment variance (std > 10)")

    # Check days scale
    if np.max(model_data.days) > 10000:
        issues.append("Very large time scale (days > 10000)")

    # Check for very short sequences
    min_seq_length = np.min(model_data.time)
    if min_seq_length < 2:
        issues.append(f"Restaurants with only {min_seq_length} review (HMM needs sequences!)")

    if issues:
        print("⚠️  Potentielle Probleme gefunden:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("✓ Keine offensichtlichen Probleme erkannt")

    print("\n" + "="*60)
    print("ZUSAMMENFASSUNG")
    print("="*60)

    if not issues and cond_number < 1e6:
        print("✓ Daten sehen gut aus für Stan!")
    elif cond_number > 1e10:
        print("⚠️  HAUPTPROBLEM: R ist extrem ill-conditioned!")
        print("   → Empfehlung: Prüfen Sie die Kovariaten auf Multikollinearität")
    else:
        print("⚠️  Einige potentielle Probleme gefunden (siehe oben)")

    return model_data

if __name__ == "__main__":
    data_path = "data/processed/processed_data.pkl"
    model_data = check_data_quality(data_path)
