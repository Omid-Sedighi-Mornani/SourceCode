import numpy as np
from scipy.stats import entropy


# calculate entropy of a given sample
def comp_entropy(x):
    # calculate probabilities
    _, counts = np.unique(x, return_counts=True)
    probabilities = counts / len(x)
    return entropy(probabilities, base=np.e)  # natural log (ln)
