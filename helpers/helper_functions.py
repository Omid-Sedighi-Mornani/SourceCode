import numpy as np
from scipy.stats import entropy
from typing import Literal, Optional
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
    if seed is None:
        # Verwende original_indices aus dem Paper
        model_path = FITTED_MODEL_FOLDER / f"{model_name}_{S}_original_indices_cmdstan.pkl"
    else:
        # Verwende spezifischen Seed
        model_path = FITTED_MODEL_FOLDER / f"{model_name}_{S}_seed{seed}_cmdstan.pkl"

    model = Model.from_pickle(model_path)
    return model
