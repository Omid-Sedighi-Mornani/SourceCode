#!/usr/bin/env python3
"""
Main script for data processing and model training.

This script performs:
1. Data processing (replicating notebook 1-processing.ipynb)
2. HMM model training (S=2,3,4)
3. VDHMM model training (S=2,3,4)
4. Multiple random seeds for consistency testing

Usage:
    python scripts/main.py [--seeds 42 123 456] [--states 2 3 4] [--models hmm vdhmm]
"""

import sys
import argparse
import pickle
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import cmdstanpy
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from constants import (
    DATA_FOLDER,
    PROCESSED_DATA_FOLDER,
    STAN_MODEL_FOLDER,
    FITTED_MODEL_FOLDER,
)
from helpers import ModelData, comp_entropy, prepare_stan_data


def process_data(seed: int = 42) -> ModelData:
    """
    Process raw data and prepare it for model training.

    Replicates the processing steps from notebook 1-processing.ipynb.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility

    Returns
    -------
    ModelData
        Processed data ready for training
    """
    print("\n" + "=" * 70)
    print("DATA PROCESSING")
    print("=" * 70)
    print(f"Random seed: {seed}")

    # Set random seed
    np.random.seed(seed)

    # Load data
    print("\n[1/8] Loading raw data...")
    reviews = pd.read_csv(DATA_FOLDER / "reviews.csv")
    business_covariates = pd.read_csv(DATA_FOLDER / "business_covariates.csv")
    print(f"  Reviews: {len(reviews)} rows")
    print(f"  Businesses: {len(business_covariates)} rows")

    # Create train/calibration/eval splits
    print("\n[2/8] Creating train/calibration/eval splits...")
    indices = np.random.permutation(len(business_covariates))
    train_indices = indices[:500]
    calibration_indices = indices[500:700]
    eval_indices = indices[700:]

    business_covariates["Train"] = 0
    business_covariates.loc[train_indices, "Train"] = 1
    business_covariates = business_covariates.sort_values(
        by="Train", ascending=False
    ).reset_index(drop=True)

    n_train = len(business_covariates["Train"])
    print(f"  Training: {len(train_indices)}")
    print(f"  Calibration: {len(calibration_indices)}")
    print(f"  Evaluation: {len(eval_indices)}")

    # Process reviews
    print("\n[3/8] Processing reviews (calculating days, ratings, sentiment)...")
    ratings = []
    sentiment = []
    days = []
    time = []
    age = []

    business_ids = business_covariates["business_id"].values
    reviews["date"] = pd.to_datetime(reviews["date"])

    for k, business_id in enumerate(business_ids):
        if k % 200 == 0:
            print(f"  Processing business {k}/{len(business_ids)}...")

        df_temp = (
            reviews[reviews["business_id"] == business_id]
            .reset_index(drop=True)
            .assign(Number=lambda x: x.index)
        )

        df_temp["Days"] = (df_temp["date"] - df_temp["date"].iloc[0]).dt.days
        days.extend(df_temp["Days"].tolist())
        sentiment.extend(df_temp["sentimenttext"].tolist())
        ratings.extend(df_temp["stars"].tolist())
        time.append(len(df_temp))
        age.append(df_temp["Days"].iloc[-1])

    print(f"  Total observations: {len(ratings)}")

    # Prepare covariates
    print("\n[4/8] Preparing covariates...")
    business_covariates["Age"] = age
    business_covariates["Checkin"] = (
        business_covariates["Checkin"] / business_covariates["Age"] * 28
    )
    business_covariates["logAge"] = np.log(business_covariates["Age"])

    relevant_covariates = business_covariates[
        [
            "density",
            "Checkin",
            "category",
            "chain",
            "Price.Level",
            "Restaurant.Size",
            "Number.of.Seats",
            "ZRI",
            "Age",
        ]
    ].copy()

    # One-hot encoding
    print("\n[5/8] One-hot encoding categorical variables...")
    relevant_covariates["category"] = relevant_covariates["category"].astype("category")
    relevant_covariates_encoded = pd.get_dummies(
        relevant_covariates, columns=["category"], drop_first=False, dtype=int
    )

    first_numeric = ["density", "Checkin"]
    category_cols = sorted(
        [
            col
            for col in relevant_covariates_encoded.columns
            if col.startswith("category_")
        ]
    )
    remaining_numeric = [
        "chain",
        "Price.Level",
        "Restaurant.Size",
        "Number.of.Seats",
        "ZRI",
        "Age",
    ]

    column_order = first_numeric + category_cols + remaining_numeric
    relevant_covariates = relevant_covariates_encoded[column_order]

    # Remove 'category_Other' (column 7, index in R is 8)
    if len(relevant_covariates.columns) > 7:
        col_to_remove = relevant_covariates.columns[7]
        if "Other" in col_to_remove:
            relevant_covariates = relevant_covariates.drop(columns=[col_to_remove])

    print(f"  Final covariate matrix: {relevant_covariates.shape}")

    # Preprocessing: Imputation and Centering
    print("\n[6/8] Preprocessing: Median imputation and centering...")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler(with_std=False)

    X_train = relevant_covariates.iloc[:n_train].copy()
    imputer.fit(X_train)
    X_train_imputed = imputer.transform(X_train)
    scaler.fit(X_train_imputed)

    cov_mat_imputed = imputer.transform(relevant_covariates)
    cov_mat_preprocessed = scaler.transform(cov_mat_imputed)

    X_train_preprocessed = cov_mat_preprocessed[:n_train]

    # QR decomposition
    print("\n[7/8] QR decomposition...")
    Q, R = np.linalg.qr(X_train_preprocessed)
    Q_scaled = Q * np.sqrt(n_train - 1)
    R_scaled = R / np.sqrt(n_train - 1)
    X_test = cov_mat_preprocessed[n_train:]

    # Prepare benchmark covariates
    print("\n[8/8] Preparing benchmark covariates...")
    review_stats = (
        reviews.groupby("business_id")
        .agg(
            VAR=("stars", "var"),
            MEAN=("stars", "mean"),
            ENTR=("stars", lambda x: comp_entropy(x)),
            COUNT=("stars", "size"),
            ONE_STAR=("stars", lambda x: (x == 1).sum()),
            TWO_STAR=("stars", lambda x: (x == 2).sum()),
            THREE_STAR=("stars", lambda x: (x == 3).sum()),
            FOUR_STAR=("stars", lambda x: (x == 4).sum()),
            FIVE_STAR=("stars", lambda x: (x == 5).sum()),
        )
        .reset_index()
    )

    for col in ["ONE_STAR", "TWO_STAR", "THREE_STAR", "FOUR_STAR", "FIVE_STAR"]:
        review_stats[col] = review_stats[col] / review_stats["COUNT"]

    benchmark_covariates = business_covariates[
        [
            "business_id",
            "density",
            "Checkin",
            "category",
            "chain",
            "Price.Level",
            "Restaurant.Size",
            "Number.of.Seats",
            "ZRI",
            "Distance.To.City.Centre",
            "Age",
            "is_open",
        ]
    ].copy()

    benchmark_covariates["Closed"] = 1 - benchmark_covariates["is_open"]
    benchmark_covariates = benchmark_covariates.merge(
        review_stats, on="business_id", how="left"
    )
    benchmark_covariates["l_COUNT"] = np.log(benchmark_covariates["COUNT"])
    benchmark_covariates["category"] = benchmark_covariates["category"].astype(
        "category"
    )
    benchmark_covariates["Closed"] = (
        benchmark_covariates["Closed"].map({1: "Closed", 0: "Open"}).astype("category")
    )

    # Create ModelData object
    model_data = ModelData(
        n_states=None,
        n_total=len(time),
        n_train=n_train,
        n_obs=int(np.sum(time)),
        n_covs=cov_mat_preprocessed.shape[1],
        time=time,
        closed=1 - business_covariates["is_open"].values,
        days=days,
        ratings=ratings,
        sentiment=sentiment,
        Q=Q_scaled,
        R=R_scaled,
        X_test=X_test,
        imputer=imputer,
        scaler=scaler,
        train_indices=train_indices,
        calibration_indices=calibration_indices,
        eval_indices=eval_indices,
        business_covariates=business_covariates,
        cov_mat=cov_mat_preprocessed,
        benchmark_covariates=benchmark_covariates,
    )

    print("\n✓ Data processing complete!")
    print(model_data.summary())

    return model_data


def train_model_cmdstan(
    model_data: ModelData,
    S: int,
    model_name: str = "hmm",
    chains: int = 4,
    parallel_chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 1000,
    seed: int = 42,
    adapt_delta: float = 0.95,
    max_treedepth: int = 10,
) -> cmdstanpy.CmdStanMCMC:
    """
    Train a model using CmdStanPy.

    Parameters
    ----------
    model_data : ModelData
        Processed data for training
    S : int
        Number of hidden states (2-5)
    model_name : str
        'vdhmm' or 'hmm'
    chains : int
        Number of MCMC chains
    parallel_chains : int
        Number of chains to run in parallel
    iter_warmup : int
        Number of warmup iterations
    iter_sampling : int
        Number of sampling iterations (post-warmup)
    seed : int
        Random seed
    adapt_delta : float
        Stan adapt_delta parameter (0.8-0.99)
    max_treedepth : int
        Stan max_treedepth parameter

    Returns
    -------
    cmdstanpy.CmdStanMCMC
        Fitted model object
    """
    assert model_name in ["vdhmm", "hmm"], f"Invalid model_name: {model_name}"
    assert S in range(2, 6), "S must be between 2 and 5"

    # Prepare Stan data
    stan_data = prepare_stan_data(model_data, S)

    # Model file
    model_file = STAN_MODEL_FOLDER / f"{model_name}.stan"
    if not model_file.exists():
        raise FileNotFoundError(f"Stan model not found: {model_file}")

    print(f"\n{'='*70}")
    print(f"Training {model_name.upper()} with S={S} states (CmdStanPy)")
    print(f"{'='*70}")
    print(f"Model file: {model_file}")
    print(f"\nConfiguration:")
    print(f"  Chains: {chains}")
    print(f"  Parallel chains: {parallel_chains}")
    print(f"  Warmup iterations: {iter_warmup}")
    print(f"  Sampling iterations: {iter_sampling}")
    print(f"  Total iterations: {iter_warmup + iter_sampling}")
    print(f"  Seed: {seed}")
    print(f"  Adapt delta: {adapt_delta}")
    print(f"  Max treedepth: {max_treedepth}")

    # Compile model
    print(f"\nCompiling model...")
    model = cmdstanpy.CmdStanModel(stan_file=str(model_file))
    print(f"✓ Model compiled")

    # Sample
    print(f"\nSampling...")
    start_time = time.time()
    fit = model.sample(
        data=stan_data,
        chains=chains,
        parallel_chains=parallel_chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        seed=seed,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        show_progress=True,
    )
    elapsed_time = time.time() - start_time

    print(f"\n✓ Sampling complete! (took {elapsed_time:.1f}s)")

    # Save model
    output_path = FITTED_MODEL_FOLDER / f"{model_name}_{S}_seed{seed}_cmdstan.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(
            {
                "fit": fit,
                "model_name": model_name,
                "S": S,
                "seed": seed,
                "stan_data": stan_data,
                "summary": fit.summary(),
                "elapsed_time": elapsed_time,
            },
            f,
        )

    print(f"✓ Model saved to {output_path}")

    # Diagnostics
    print(f"\n{'-'*70}")
    print("Diagnostics:")
    print(f"{'-'*70}")
    print(fit.diagnose())

    # Summary statistics
    print(f"\n{'-'*70}")
    print("Summary (first 20 parameters):")
    print(f"{'-'*70}")
    summary_df = fit.summary()
    print(summary_df.head(20))

    return fit


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Data processing and model training pipeline"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Random seeds to test (default: 42 123 456)",
    )
    parser.add_argument(
        "--states",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="Number of hidden states to test (default: 2 3 4)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["hmm", "vdhmm"],
        choices=["hmm", "vdhmm"],
        help="Models to train (default: hmm vdhmm)",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of MCMC chains (default: 4)",
    )
    parser.add_argument(
        "--parallel-chains",
        type=int,
        default=4,
        help="Number of parallel chains (default: 4)",
    )
    parser.add_argument(
        "--iter-warmup",
        type=int,
        default=1000,
        help="Number of warmup iterations (default: 1000)",
    )
    parser.add_argument(
        "--iter-sampling",
        type=int,
        default=1000,
        help="Number of sampling iterations (default: 1000)",
    )
    parser.add_argument(
        "--adapt-delta",
        type=float,
        default=0.95,
        help="Stan adapt_delta parameter (default: 0.95)",
    )
    parser.add_argument(
        "--max-treedepth",
        type=int,
        default=10,
        help="Stan max_treedepth parameter (default: 10)",
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip data processing and load from processed_data.pkl",
    )

    args = parser.parse_args()

    # Create output directories
    PROCESSED_DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    FITTED_MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("HMM & VDHMM TRAINING PIPELINE")
    print("=" * 70)
    print(f"\nCmdStanPy Version: {cmdstanpy.__version__}")
    print(f"CmdStan Path: {cmdstanpy.cmdstan_path()}")
    print(f"\nConfiguration:")
    print(f"  Random seeds: {args.seeds}")
    print(f"  Hidden states: {args.states}")
    print(f"  Models: {args.models}")
    print(f"  Chains: {args.chains}")
    print(f"  Parallel chains: {args.parallel_chains}")
    print(f"  Warmup iterations: {args.iter_warmup}")
    print(f"  Sampling iterations: {args.iter_sampling}")
    print(f"  Adapt delta: {args.adapt_delta}")
    print(f"  Max treedepth: {args.max_treedepth}")

    # Track all trained models
    trained_models = {}
    processing_times = {}

    # Loop over seeds
    for seed_idx, seed in enumerate(args.seeds):
        print("\n" + "=" * 70)
        print(f"SEED {seed_idx + 1}/{len(args.seeds)}: {seed}")
        print("=" * 70)

        # Process data or load existing
        if args.skip_processing and seed == args.seeds[0]:
            print("\n[Skipping data processing, loading from file...]")
            processed_data_path = PROCESSED_DATA_FOLDER / "processed_data.pkl"
            if not processed_data_path.exists():
                raise FileNotFoundError(
                    f"Processed data not found: {processed_data_path}\n"
                    "Run without --skip-processing first."
                )
            model_data = ModelData.from_pickle(processed_data_path)
            print(model_data.summary())
        else:
            model_data = process_data(seed=seed)

            # Save processed data for first seed
            if seed_idx == 0:
                output_path = PROCESSED_DATA_FOLDER / "processed_data.pkl"
                model_data.to_pickle(output_path)
                print(f"\n✓ Processed data saved to {output_path}")

        # Train models
        for model_name in args.models:
            for S in args.states:
                model_key = f"{model_name}_S{S}_seed{seed}"

                try:
                    print(f"\n\n{'#'*70}")
                    print(f"# {model_name.upper()} Training: S={S}, Seed={seed}")
                    print(f"{'#'*70}\n")

                    fit = train_model_cmdstan(
                        model_data=model_data,
                        S=S,
                        model_name=model_name,
                        chains=args.chains,
                        parallel_chains=args.parallel_chains,
                        iter_warmup=args.iter_warmup,
                        iter_sampling=args.iter_sampling,
                        seed=seed,
                        adapt_delta=args.adapt_delta,
                        max_treedepth=args.max_treedepth,
                    )

                    trained_models[model_key] = fit
                    print(
                        f"\n✓✓✓ {model_name.upper()} with S={S}, Seed={seed} completed! ✓✓✓\n"
                    )

                except Exception as e:
                    print(
                        f"\n✗✗✗ Error training {model_name.upper()} with S={S}, Seed={seed}: {e} ✗✗✗\n"
                    )
                    raise

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"\nTotal models trained: {len(trained_models)}")
    print(f"\nModels trained:")
    for model_key in sorted(trained_models.keys()):
        print(f"  - {model_key}")

    print(f"\nSaved in: {FITTED_MODEL_FOLDER}")

    # List all saved models
    saved_models = sorted(FITTED_MODEL_FOLDER.glob("*_cmdstan.pkl"))
    print(f"\nAll saved model files ({len(saved_models)}):")
    for model_file in saved_models:
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  - {model_file.name} ({size_mb:.2f} MB)")

    print("\n" + "=" * 70)
    print("ALL TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
