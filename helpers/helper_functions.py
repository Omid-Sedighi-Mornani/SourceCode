import numpy as np
from scipy.stats import entropy
from .model_data import ModelData


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
