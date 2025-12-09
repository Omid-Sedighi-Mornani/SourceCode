import numpy as np
import pandas as pd
from scipy.stats import entropy
from typing import List, Literal, Optional
from .model_data import ModelData
from .model import Model
from constants import FITTED_MODEL_FOLDER


# calculate entropy of a given sample
def comp_entropy(x):
    # calculate probabilities
    _, counts = np.unique(x, return_counts=True)
    probabilities = counts / len(x)
    return entropy(probabilities, base=np.e)  # natural log (ln)


def prepare_stan_data(model_data: ModelData, S: int):
    """
    Bereitet die Daten für Stan vor.

    CmdStanPy braucht die Daten als JSON-kompatibles Dictionary.
    """
    stan_data = {
        "S": S,
        "N_total": int(model_data.n_total),
        "N_train": int(model_data.n_train),
        "N_obs": int(model_data.n_obs),
        "nCovs": int(model_data.n_covs),
        "Time": [int(x) for x in model_data.time],
        "Closed": [int(x) for x in model_data.closed],
        "Days": [float(x) for x in model_data.days],
        "Ratings": [int(x) for x in model_data.ratings],
        "Sentiment": [float(x) for x in model_data.sentiment],
        "Q": model_data.Q.tolist(),
        "R": model_data.R.tolist(),
        "X_test": model_data.X_test.tolist(),
    }
    return stan_data


def load_fitted_model(
    model_name: Literal["hmm", "vdhmm"],
    S: int,
    seed: Optional[int] = None,
    model_folder=FITTED_MODEL_FOLDER,
) -> Model:
    """
    Lädt ein trainiertes CmdStanPy Modell.

    Parameters:
    -----------
    model_name : Literal["hmm", "vdhmm"]
        Name des Modells ('hmm' oder 'vdhmm')
    S : int
        Anzahl der Hidden States (2-5)
    seed : Optional[int], default=None
        Random seed des trainierten Modells.
        - Wenn seed angegeben ist: Lädt Modell mit diesem Seed
        - Wenn seed=None: Lädt Modell mit original_indices aus dem Paper

    Returns:
    --------
    Model
        Geladenes Modell-Objekt

    Examples:
    ---------
    >>> # Lade Modell mit Seed 42
    >>> model = load_fitted_model("hmm", 3, seed=42)

    >>> # Lade Modell mit Original-Indices aus dem Paper
    >>> model = load_fitted_model("vdhmm", 4, seed=None)
    """
    if model_folder is None:
        model_folder = FITTED_MODEL_FOLDER

    if seed is None:
        # Verwende original_indices aus dem Paper
        model_path = model_folder / f"{model_name}_{S}_original_indices_cmdstan.pkl"
    else:
        # Verwende spezifischen Seed
        model_path = model_folder / f"{model_name}_{S}_seed{seed}_cmdstan.pkl"

    model = Model.from_pickle(model_path)
    return model


def build_summary_df(
    ratings: List[int],
    temp_diff: List[int],
    sentiment: List[float],
    time: List[int],
    business_covariates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Builds a summary statistics DataFrame with mean and standard deviation for 12 key variables.

    Calculates descriptive statistics (mean and SD) for review-level data (ratings, sentiment,
    interarrival times) and business-level covariates (density, age, check-in rate, etc.).

    Parameters:
    -----------
    ratings : List[int]
        Star ratings for reviews (1-5 scale)
    temp_diff : List[int]
        Time differences between consecutive ratings (in days), with negative values removed
    sentiment : List[float]
        Sentiment scores for reviews
    time : List[int]
        Number of reviews per business
    business_covariates : pd.DataFrame
        DataFrame containing business-level covariates with columns:
        - density: Restaurant density in area
        - Age: Business age (in days, converted to months by dividing by 28)
        - Checkin: Check-in rate
        - chain: Binary indicator for chain status
        - ZRI: Zillow Rent Index (rent level)
        - Restaurant.Size: Size in m²
        - Number.of.Seats: Seating capacity
        - Closed: Binary indicator (1=closed, 0=open)

    Returns:
    --------
    pd.DataFrame
        Summary statistics with columns ['Variable', 'Mean', 'SD'] containing 12 rows:
        1. Rating
        2. Days Between Ratings
        3. Sentiment
        4. Density
        5. Age (in months)
        6. Check-in rate
        7. Chain status
        8. Rent level (Zillow Rent Index)
        9. Restaurant Size (in m^2)
        10. Number of Seats
        11. Time
        12. Closed
    """
    summary_data = {"Variable": [], "Mean": [], "SD": []}

    # Rating level
    summary_data["Variable"].append("Rating")
    summary_data["Mean"].append(round(np.mean(ratings), 2))
    summary_data["SD"].append(round(np.std(ratings, ddof=1), 2))

    # Days Between Ratings
    summary_data["Variable"].append("Days Between Ratings")
    summary_data["Mean"].append(round(np.mean(temp_diff), 2))
    summary_data["SD"].append(round(np.std(temp_diff, ddof=1), 2))

    # Sentiment statistics
    summary_data["Variable"].append("Sentiment")
    summary_data["Mean"].append(round(np.nanmean(sentiment), 2))
    summary_data["SD"].append(round(np.nanstd(sentiment, ddof=1), 2))

    # Restaurant Density
    summary_data["Variable"].append("Density")
    summary_data["Mean"].append(round(np.mean(business_covariates["density"]), 2))
    summary_data["SD"].append(round(np.std(business_covariates["density"], ddof=1), 2))

    # Age (in Monaten, daher / 28)
    summary_data["Variable"].append("Age (in months)")
    summary_data["Mean"].append(round(np.mean(business_covariates["Age"]) / 28, 2))
    summary_data["SD"].append(round(np.std(business_covariates["Age"] / 28, ddof=1), 2))

    # Checkin
    summary_data["Variable"].append("Check-in rate")
    summary_data["Mean"].append(round(np.mean(business_covariates["Checkin"]), 2))
    summary_data["SD"].append(round(np.std(business_covariates["Checkin"], ddof=1), 2))

    # Chain status
    summary_data["Variable"].append("Chain status")
    summary_data["Mean"].append(round(np.mean(business_covariates["chain"]), 2))
    summary_data["SD"].append(round(np.std(business_covariates["chain"], ddof=1), 2))

    # ZRI (Rent Level)
    summary_data["Variable"].append("Rent level (Zillow Rent Index)")
    summary_data["Mean"].append(round(np.nanmean(business_covariates["ZRI"]), 2))
    summary_data["SD"].append(round(np.nanstd(business_covariates["ZRI"], ddof=1), 2))

    # Restaurant Size
    summary_data["Variable"].append("Restaurant Size (in m^2)")
    summary_data["Mean"].append(
        round(np.nanmean(business_covariates["Restaurant.Size"]), 2)
    )
    summary_data["SD"].append(
        round(np.nanstd(business_covariates["Restaurant.Size"], ddof=1), 2)
    )

    # Number of Seats
    summary_data["Variable"].append("Number of Seats")
    summary_data["Mean"].append(
        round(np.nanmean(business_covariates["Number.of.Seats"]), 2)
    )
    summary_data["SD"].append(
        round(np.nanstd(business_covariates["Number.of.Seats"], ddof=1), 2)
    )

    # Time
    summary_data["Variable"].append("Time")
    summary_data["Mean"].append(round(np.mean(time), 2))
    summary_data["SD"].append(round(np.std(time, ddof=1), 2))

    # Closed
    summary_data["Variable"].append("Closed")
    summary_data["Mean"].append(round(np.mean(business_covariates["Closed"]), 2))
    summary_data["SD"].append(round(np.std(business_covariates["Closed"], ddof=1), 2))

    summary_df = pd.DataFrame(summary_data)

    return summary_df
