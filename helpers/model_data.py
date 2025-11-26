"""
Data classes for managing model data with type safety and serialization.
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelData:
    """
    Container for all model data including raw data, processed arrays,
    preprocessed matrices, preprocessing artifacts, and benchmark data.

    Attributes:
        n_states: Number of states (for HMM models, None if not applicable)
        n_total: Total number of businesses
        n_train: Number of training samples
        n_obs: Total number of observations
        n_covs: Number of covariates
        time: List of time points for each business
        closed: Array indicating if business is closed (1) or open (0)
        days: Days since first review for each observation
        ratings: Star ratings for each review
        sentiment: Sentiment scores for each review
        Q: Q matrix from QR decomposition (scaled)
        R: R matrix from QR decomposition (scaled)
        X_test: Test covariate matrix
        imputer: SimpleImputer for handling missing values
        scaler: StandardScaler for scaling features
        train_indices: Indices for training set
        calibration_indices: Indices for calibration set
        eval_indices: Indices for evaluation set
        business_covariates: Original business covariates DataFrame
        cov_mat: Preprocessed covariate matrix
        benchmark_covariates: DataFrame with aggregated statistics for benchmark models
    """

    # Metadata
    n_states: Optional[int] = None
    n_total: int = 0
    n_train: int = 0
    n_obs: int = 0
    n_covs: int = 0

    # Time series data
    time: list[int] = field(default_factory=list)
    closed: np.ndarray = field(default_factory=lambda: np.array([]))
    days: list[int] = field(default_factory=list)

    # Review data
    ratings: list[int] = field(default_factory=list)
    sentiment: list[float] = field(default_factory=list)

    # Preprocessed matrices
    Q: np.ndarray = field(default_factory=lambda: np.array([]))
    R: np.ndarray = field(default_factory=lambda: np.array([]))
    X_test: np.ndarray = field(default_factory=lambda: np.array([]))

    # Preprocessing artifacts
    imputer: Optional[SimpleImputer] = None
    scaler: Optional[StandardScaler] = None
    train_indices: Optional[np.ndarray] = None
    calibration_indices: Optional[np.ndarray] = None
    eval_indices: Optional[np.ndarray] = None
    business_covariates: Optional[pd.DataFrame] = None
    cov_mat: Optional[np.ndarray] = None

    # Benchmark data
    benchmark_covariates: Optional[pd.DataFrame] = None

    def __post_init__(self):
        """Validate data after initialization."""
        if self.n_obs > 0:
            assert len(self.ratings) == self.n_obs, "Ratings length mismatch"
            assert len(self.sentiment) == self.n_obs, "Sentiment length mismatch"
            assert len(self.days) == self.n_obs, "Days length mismatch"

        if self.n_total > 0:
            assert len(self.time) == self.n_total, "Time length mismatch"
            assert len(self.closed) == self.n_total, "Closed length mismatch"

    @classmethod
    def from_dict(cls, data: dict) -> "ModelData":
        """
        Create ModelData instance from dictionary.

        Args:
            data: Dictionary containing model data

        Returns:
            ModelData instance
        """
        # Handle nested 'data' key for backwards compatibility
        model_dict = data.get("data", data)

        return cls(
            n_states=model_dict.get("n_states") or model_dict.get("S"),
            n_total=(
                model_dict.get("N_total", 0)
                if "N_total" in model_dict
                else model_dict.get("n_total", 0)
            ),
            n_train=model_dict.get("n_train", 0),
            n_obs=(
                model_dict.get("n_obs", 0)
                if "n_obs" in model_dict
                else model_dict.get("N_obs", 0)
            ),
            n_covs=model_dict.get("n_covs", 0),
            time=model_dict.get("time", []),
            closed=model_dict.get("closed", np.array([])),
            days=model_dict.get("days", []),
            ratings=model_dict.get("ratings", []),
            sentiment=model_dict.get("sentiment", []),
            Q=model_dict.get("Q", np.array([])),
            R=model_dict.get("R", np.array([])),
            X_test=model_dict.get("X_test", np.array([])),
            imputer=data.get("imputer"),
            scaler=data.get("scaler"),
            train_indices=data.get("train_indices"),
            calibration_indices=data.get("calibration_indices"),
            eval_indices=data.get("eval_indices"),
            business_covariates=data.get("business_covariates"),
            cov_mat=data.get("cov_mat"),
            benchmark_covariates=data.get("benchmark_covariates"),
        )

    def to_dict(self) -> dict:
        """
        Convert ModelData to dictionary.

        Returns:
            Dictionary representation of ModelData
        """
        return {
            "data": {
                "n_states": self.n_states,
                "n_total": self.n_total,
                "n_train": self.n_train,
                "n_obs": self.n_obs,
                "n_covs": self.n_covs,
                "time": self.time,
                "closed": self.closed,
                "days": self.days,
                "ratings": self.ratings,
                "sentiment": self.sentiment,
                "Q": self.Q,
                "R": self.R,
                "X_test": self.X_test,
            },
            "imputer": self.imputer,
            "scaler": self.scaler,
            "train_indices": self.train_indices,
            "calibration_indices": self.calibration_indices,
            "eval_indices": self.eval_indices,
            "business_covariates": self.business_covariates,
            "cov_mat": self.cov_mat,
            "benchmark_covariates": self.benchmark_covariates,
        }

    @classmethod
    def from_pickle(cls, filepath: Union[str, Path]) -> "ModelData":
        """
        Load ModelData from pickle file.

        Args:
            filepath: Path to pickle file

        Returns:
            ModelData instance
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Pickle file not found: {filepath}")

        with open(filepath, "rb") as f:
            saved_data = pickle.load(f)

        return cls.from_dict(saved_data)

    def to_pickle(self, filepath: Union[str, Path]) -> None:
        """
        Save ModelData to pickle file.

        Args:
            filepath: Path where to save pickle file
        """
        filepath = Path(filepath)

        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self.to_dict(), f)

    def summary(self) -> str:
        """
        Get a summary of the ModelData.

        Returns:
            String summary of the data
        """
        has_preprocessing = self.imputer is not None and self.scaler is not None
        has_benchmark = self.benchmark_covariates is not None

        summary_str = f"""
ModelData Summary:
==================
States (HMM): {self.n_states if self.n_states else 'Not set'}
Total businesses: {self.n_total}
Training samples: {self.n_train}
Total observations: {self.n_obs}
Number of covariates: {self.n_covs}

Data shapes:
- Ratings: {len(self.ratings)}
- Sentiment: {len(self.sentiment)}
- Days: {len(self.days)}
- Q matrix: {self.Q.shape if self.Q.size > 0 else 'empty'}
- R matrix: {self.R.shape if self.R.size > 0 else 'empty'}
- X_test: {self.X_test.shape if self.X_test.size > 0 else 'empty'}

Business status:
- Closed: {np.sum(self.closed) if self.closed.size > 0 else 0}
- Open: {np.sum(1 - self.closed) if self.closed.size > 0 else 0}

Preprocessing artifacts: {'Available' if has_preprocessing else 'Not available'}
"""
        if has_preprocessing:
            summary_str += f"""- Train indices: {len(self.train_indices) if self.train_indices is not None else 0}
- Calibration indices: {len(self.calibration_indices) if self.calibration_indices is not None else 0}
- Eval indices: {len(self.eval_indices) if self.eval_indices is not None else 0}
"""

        if has_benchmark:
            summary_str += f"""
Benchmark data: Available
- Benchmark covariates shape: {self.benchmark_covariates.shape}
- Columns: {', '.join(self.benchmark_covariates.columns.tolist()[:5])}{'...' if len(self.benchmark_covariates.columns) > 5 else ''}
"""
        else:
            summary_str += "\nBenchmark data: Not available\n"

        return summary_str

    def __repr__(self) -> str:
        """String representation of ModelData."""
        return f"ModelData(n_total={self.n_total}, n_train={self.n_train}, n_obs={self.n_obs})"
