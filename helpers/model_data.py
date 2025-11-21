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
class PreprocessingArtifacts:
    """
    Container for preprocessing artifacts (imputer, scaler, indices).
    """

    imputer: SimpleImputer
    scaler: StandardScaler
    train_indices: np.ndarray
    calibration_indices: np.ndarray
    eval_indices: np.ndarray
    business_covariates: pd.DataFrame
    cov_mat: np.ndarray

    @classmethod
    def from_dict(cls, data: dict) -> "PreprocessingArtifacts":
        """Create PreprocessingArtifacts from dictionary."""
        return cls(
            imputer=data["imputer"],
            scaler=data["scaler"],
            train_indices=data["train_indices"],
            calibration_indices=data["calibration_indices"],
            eval_indices=data["eval_indices"],
            business_covariates=data["business_covariates"],
            cov_mat=data["cov_mat"],
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "imputer": self.imputer,
            "scaler": self.scaler,
            "train_indices": self.train_indices,
            "calibration_indices": self.calibration_indices,
            "eval_indices": self.eval_indices,
            "business_covariates": self.business_covariates,
            "cov_mat": self.cov_mat,
        }


@dataclass
class ModelData:
    """
    Container for all model data including raw data, processed arrays,
    and preprocessed matrices.

    Attributes:
        S: Optional parameter (None in Python corresponds to NA in R)
        N_total: Total number of businesses
        N_train: Number of training samples
        N_obs: Total number of observations
        n_covs: Number of covariates
        time: List of time points for each business
        closed: Array indicating if business is closed (1) or open (0)
        days: Days since first review for each observation
        ratings: Star ratings for each review
        sentiment: Sentiment scores for each review
        Q: Q matrix from QR decomposition (scaled)
        R: R matrix from QR decomposition (scaled)
        X_test: Test covariate matrix
    """

    # Metadata
    S: Optional[int] = None
    N_total: int = 0
    N_train: int = 0
    N_obs: int = 0
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

    def __post_init__(self):
        """Validate data after initialization."""
        if self.N_obs > 0:
            assert len(self.ratings) == self.N_obs, "Ratings length mismatch"
            assert len(self.sentiment) == self.N_obs, "Sentiment length mismatch"
            assert len(self.days) == self.N_obs, "Days length mismatch"

        if self.N_total > 0:
            assert len(self.time) == self.N_total, "Time length mismatch"
            assert len(self.closed) == self.N_total, "Closed length mismatch"

    @classmethod
    def from_dict(cls, data: dict) -> "ModelData":
        """
        Create ModelData instance from dictionary.

        Args:
            data: Dictionary containing model data

        Returns:
            ModelData instance
        """
        return cls(
            S=data.get("S"),
            N_total=data.get("N_total", 0),
            N_train=data.get("N_train", 0),
            N_obs=data.get("N_obs", 0),
            n_covs=data.get("n_covs", 0),
            time=data.get("time", []),
            closed=data.get("closed", np.array([])),
            days=data.get("days", []),
            ratings=data.get("ratings", []),
            sentiment=data.get("sentiment", []),
            Q=data.get("Q", np.array([])),
            R=data.get("R", np.array([])),
            X_test=data.get("X_test", np.array([])),
        )

    def to_dict(self) -> dict:
        """
        Convert ModelData to dictionary.

        Returns:
            Dictionary representation of ModelData
        """
        return {
            "S": self.S,
            "N_total": self.N_total,
            "N_train": self.N_train,
            "N_obs": self.N_obs,
            "n_covs": self.n_covs,
            "time": self.time,
            "closed": self.closed,
            "days": self.days,
            "ratings": self.ratings,
            "sentiment": self.sentiment,
            "Q": self.Q,
            "R": self.R,
            "X_test": self.X_test,
        }

    @classmethod
    def from_pickle(
        cls, filepath: Union[str, Path]
    ) -> tuple["ModelData", "PreprocessingArtifacts"]:
        """
        Load ModelData and PreprocessingArtifacts from pickle file.

        Args:
            filepath: Path to pickle file

        Returns:
            Tuple of (ModelData, PreprocessingArtifacts)
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Pickle file not found: {filepath}")

        with open(filepath, "rb") as f:
            saved_data = pickle.load(f)

        model_data = cls.from_dict(saved_data["data"])
        preprocessing_artifacts = PreprocessingArtifacts.from_dict(saved_data)

        return model_data, preprocessing_artifacts

    def to_pickle(
        self,
        filepath: Union[str, Path],
        preprocessing_artifacts: Optional[PreprocessingArtifacts] = None,
    ) -> None:
        """
        Save ModelData and optional PreprocessingArtifacts to pickle file.

        Args:
            filepath: Path where to save pickle file
            preprocessing_artifacts: Optional preprocessing artifacts to save
        """
        filepath = Path(filepath)

        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data_to_save = {"data": self.to_dict()}

        if preprocessing_artifacts is not None:
            data_to_save.update(preprocessing_artifacts.to_dict())

        with open(filepath, "wb") as f:
            pickle.dump(data_to_save, f)

    def summary(self) -> str:
        """
        Get a summary of the ModelData.

        Returns:
            String summary of the data
        """
        return f"""
ModelData Summary:
==================
Total businesses: {self.N_total}
Training samples: {self.N_train}
Total observations: {self.N_obs}
Number of covariates: {self.n_covs}

Data shapes:
- Ratings: {len(self.ratings)}
- Sentiment: {len(self.sentiment)}
- Days: {len(self.days)}
- Q matrix: {self.Q.shape if self.Q.size > 0 else 'empty'}
- R matrix: {self.R.shape if self.R.size > 0 else 'empty'}
- X_test: {self.X_test.shape if self.X_test.size > 0 else 'empty'}

Business status:
- Closed: {np.sum(self.closed)}
- Open: {np.sum(1 - self.closed)}
"""

    def __repr__(self) -> str:
        """String representation of ModelData."""
        return f"ModelData(N_total={self.N_total}, N_train={self.N_train}, N_obs={self.N_obs})"
