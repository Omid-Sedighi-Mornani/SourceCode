"""
Data classes for managing trained models with type safety and serialization.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd
from cmdstanpy import CmdStanMCMC


@dataclass
class Model:
    """
    Container for trained HMM/VD-HMM models including fit object, metadata, and summary statistics.

    Attributes:
        fit: CmdStanMCMC fit object from cmdstanpy
        model_name: Type of model ('vdhmm' or 'hmm')
        S: Number of hidden states
        stan_data: Dictionary containing the data used for Stan model fitting
        summary: DataFrame with summary statistics of model parameters
        chains: Number of MCMC chains used
        iter_warmup: Number of warmup iterations
        iter_sampling: Number of sampling iterations
        seed: Random seed used for training
        adapt_delta: Stan adapt_delta parameter used
        max_treedepth: Stan max_treedepth parameter used
    """

    # Core model components
    fit: CmdStanMCMC
    model_name: Literal["vdhmm", "hmm"]
    S: int
    stan_data: dict
    summary: pd.DataFrame

    # Training configuration (optional metadata)
    chains: Optional[int] = None
    iter_warmup: Optional[int] = None
    iter_sampling: Optional[int] = None
    seed: Optional[int] = None
    adapt_delta: Optional[float] = None
    max_treedepth: Optional[int] = None

    def __post_init__(self):
        """Validate model after initialization."""
        assert self.model_name in [
            "vdhmm",
            "hmm_parallelized",
            "vdhmm_parallelized",
            "hmm",
        ], f"Invalid model_name: {self.model_name}"
        assert self.S in range(2, 6), f"S must be between 2 and 5, got {self.S}"
        assert isinstance(
            self.fit, CmdStanMCMC
        ), f"fit must be CmdStanMCMC object, got {type(self.fit)}"
        assert isinstance(
            self.summary, pd.DataFrame
        ), f"summary must be DataFrame, got {type(self.summary)}"

    @classmethod
    def from_dict(cls, data: dict) -> "Model":
        """
        Create Model instance from dictionary.

        Args:
            data: Dictionary containing model data

        Returns:
            Model instance
        """
        return cls(
            fit=data["fit"],
            model_name=data["model_name"],
            S=data["S"],
            stan_data=data["stan_data"],
            summary=data["summary"],
            chains=data.get("chains"),
            iter_warmup=data.get("iter_warmup"),
            iter_sampling=data.get("iter_sampling"),
            seed=data.get("seed"),
            adapt_delta=data.get("adapt_delta"),
            max_treedepth=data.get("max_treedepth"),
        )

    def to_dict(self) -> dict:
        """
        Convert Model to dictionary.

        Returns:
            Dictionary representation of Model
        """
        result = {
            "fit": self.fit,
            "model_name": self.model_name,
            "S": self.S,
            "stan_data": self.stan_data,
            "summary": self.summary,
        }

        # Add optional training configuration if available
        if self.chains is not None:
            result["chains"] = self.chains
        if self.iter_warmup is not None:
            result["iter_warmup"] = self.iter_warmup
        if self.iter_sampling is not None:
            result["iter_sampling"] = self.iter_sampling
        if self.seed is not None:
            result["seed"] = self.seed
        if self.adapt_delta is not None:
            result["adapt_delta"] = self.adapt_delta
        if self.max_treedepth is not None:
            result["max_treedepth"] = self.max_treedepth

        return result

    @classmethod
    def from_pickle(cls, filepath: Union[str, Path]) -> "Model":
        """
        Load Model from pickle file.

        Args:
            filepath: Path to pickle file

        Returns:
            Model instance
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Pickle file not found: {filepath}")

        with open(filepath, "rb") as f:
            saved_data = pickle.load(f)

        return cls.from_dict(saved_data)

    def to_pickle(self, filepath: Union[str, Path]) -> None:
        """
        Save Model to pickle file.

        Args:
            filepath: Path where to save pickle file
        """
        filepath = Path(filepath)

        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self.to_dict(), f)

    def diagnose(self) -> str:
        """
        Run diagnostics on the fitted model.

        Returns:
            Diagnostic output as string
        """
        return self.fit.diagnose()

    def get_parameter_summary(
        self, parameters: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Get summary statistics for specific parameters.

        Args:
            parameters: List of parameter names to include. If None, returns all.

        Returns:
            DataFrame with parameter statistics
        """
        if parameters is None:
            return self.summary

        # Filter summary for specified parameters
        return self.summary[self.summary.index.str.startswith(tuple(parameters))]

    def summary_text(self) -> str:
        """
        Get a text summary of the Model.

        Returns:
            String summary of the model
        """
        n_params = len(self.summary)
        total_samples = (
            self.chains * self.iter_sampling
            if self.chains and self.iter_sampling
            else "Unknown"
        )

        summary_str = f"""
Model Summary:
==============
Model type: {self.model_name.upper()}
Number of states (S): {self.S}
Total parameters: {n_params}

Training Configuration:
-----------------------
"""
        if self.chains is not None:
            summary_str += f"Chains: {self.chains}\n"
        if self.iter_warmup is not None:
            summary_str += f"Warmup iterations: {self.iter_warmup}\n"
        if self.iter_sampling is not None:
            summary_str += f"Sampling iterations: {self.iter_sampling}\n"
        if self.chains and self.iter_sampling:
            summary_str += f"Total posterior samples: {total_samples}\n"
        if self.seed is not None:
            summary_str += f"Seed: {self.seed}\n"
        if self.adapt_delta is not None:
            summary_str += f"Adapt delta: {self.adapt_delta}\n"
        if self.max_treedepth is not None:
            summary_str += f"Max treedepth: {self.max_treedepth}\n"

        summary_str += f"""
Stan Data:
----------
Keys: {list(self.stan_data.keys())}
N_total: {self.stan_data.get('N_total', 'N/A')}
N_train: {self.stan_data.get('N_train', 'N/A')}
N_obs: {self.stan_data.get('N_obs', 'N/A')}
nCovs: {self.stan_data.get('nCovs', 'N/A')}

Parameter Summary (first 10):
-----------------------------
{self.summary.head(10).to_string()}
"""
        return summary_str

    def __repr__(self) -> str:
        """String representation of Model."""
        return f"Model(model_name='{self.model_name}', S={self.S}, n_params={len(self.summary)})"
