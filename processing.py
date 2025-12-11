# %% [markdown]
# ## Data Processing – Replicating “I Will Survive: Predicting Business Failures from Customer Ratings”
#
# The following pipeline mirrors the data preparation and analysis steps of the Marketing Science case study and stores them in a Pickle file for further analysis.
#

# %%
import sys
from pathlib import Path

# add projects root directory to the system path to enable importing custom modules (e.g., from the "helpers" folder).
sys.path.append(str(Path("..").resolve()))

# Imports
import pandas as pd
import numpy as np


SEED = 42  # random seed for reproducability (change seed, if desired)
SET_ORIGINAL_INDICES = False  # if set to true, the original paper indices are selected

np.random.seed(SEED)

# %%
from constants import DATA_FOLDER

# Dataframes
reviews = pd.read_csv(DATA_FOLDER / "reviews.csv")
business_covariates = pd.read_csv(DATA_FOLDER / "business_covariates.csv")

# %%
business_covariates.columns

# %%
# create indices for training evaluation and calibration

from constants import CALIBRATION_INDICES, EVAL_INDICES, TRAIN_INDICES

# get original indices
if SET_ORIGINAL_INDICES:
    train_indices = TRAIN_INDICES
    calibration_indices = CALIBRATION_INDICES
    eval_indices = EVAL_INDICES

# get indices based on random seed
else:
    indices = np.random.permutation(len(business_covariates))
    indices_val_cal = np.random.permutation(
        np.arange(500, len(business_covariates))
    )  # range 500-921 (because of sorting)

    train_indices = indices[:500]  # take 500 random samples
    calibration_indices = indices_val_cal[:100]  # take 100 random out of range 500-921
    eval_indices = indices_val_cal[100:]  # take 321 random out range 500-921

# %%
assert (business_covariates.get("TRAIN")).sum() == 0, "training set already assigned!"

# set 'TRAIN' variable to 1 for train_indices, 0 otherwise
business_covariates.loc[train_indices, "TRAIN"] = 1

# sort business_covariates so that rows with Train==1 come first
business_covariates = business_covariates.sort_values(
    by="TRAIN", ascending=False
).reset_index(
    drop=True
)  # it is possible to retreive all training data with :500

n_train = len(train_indices)  # number of training samples (500)

print(f"Train/Calibration/Eval indices created:")
print(f"  Train: {len(train_indices)} samples")
print(f"  Calibration: {len(calibration_indices)} samples")
print(f"  Eval: {len(eval_indices)} samples")

# %%
# initialize list with data needed for stan

ratings = []
sentiment = []
days = []
time = []
age = []

business_ids = business_covariates["business_id"].values

# convert date string into datetime object
reviews["date"] = pd.to_datetime(reviews["date"])

for k, business_id in enumerate(business_ids):
    if k % 100 == 0:
        print(f"[{k}] - Conversion for business_id: {business_id}")

    # get temporary dataframe of all reviews with given business_id and assign column 'Number' (Rating 0, ..., M_i)
    df_temp = (
        reviews[reviews["business_id"] == business_id]
        .reset_index(drop=True)
        .assign(Number=lambda x: x.index)
    )

    # calculate days since first review
    df_temp["Days"] = (df_temp["date"] - df_temp["date"].iloc[0]).dt.days
    days.extend(df_temp["Days"].tolist())
    sentiment.extend(df_temp["sentimenttext"].tolist())
    ratings.extend(df_temp["stars"].tolist())
    time.append(len(df_temp))
    age.append(df_temp["Days"].iloc[-1])


# validation checks
assert sum(time) == len(sentiment)
assert len(days) == len(sentiment)
assert len(ratings) == len(sentiment)
print("Done ... validation checks passed!")
print("Created required lists for MCMC sampling")

# %%
assert (
    not "Age" in business_covariates
), "The key Age is already added to dataframe! Make sure, that you only run this cell once!"

# add restaurant age in days to dataframe
business_covariates["Age"] = age

# change checkin count to checkin rates (number of checkins every month, assuming a month contains 28 days)
business_covariates["Checkin"] = (
    business_covariates["Checkin"] / business_covariates["Age"] * 28
)

# add log of age to dataframe for later analysis
business_covariates["logAge"] = np.log(business_covariates["Age"])

# only get relevant covariates for training
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

relevant_covariates

# %%
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

assert (
    len(relevant_covariates.columns) == 9
), "One Hot Coding and Scaling already performed on dataframe!"

# Mark as categorial variable
relevant_covariates["category"] = relevant_covariates["category"].astype("category")

# One Hot Encoding
relevant_covariates_encoded = pd.get_dummies(
    relevant_covariates, columns=["category"], drop_first=False, dtype=int
)

## Create model matrix (cov_mat)

# First two numeric columns before category column
first_numeric = ["density", "Checkin"]

# Category dummies (alphabetically sorted)
category_cols = sorted(
    [col for col in relevant_covariates_encoded.columns if col.startswith("category_")]
)

# Remaining numeric columns after category in original order
remaining_numeric = [
    "chain",
    "Price.Level",
    "Restaurant.Size",
    "Number.of.Seats",
    "ZRI",
    "Age",
]

# combine original order
column_order = first_numeric + category_cols + remaining_numeric
relevant_covariates = relevant_covariates_encoded[column_order]

# remove category_Other (8th column)
if len(relevant_covariates.columns) > 7:
    col_to_remove = relevant_covariates.columns[7]
    print(f"Removing column at index 7 (category_Other): '{col_to_remove}'")

    # Verify it's categoryOther
    if "Other" in col_to_remove:
        relevant_covariates = relevant_covariates.drop(columns=[col_to_remove])
    else:
        # if not expected
        print(f"WARNING: Expected 'categoryOther' but found '{col_to_remove}'")
        print(f"All columns: {relevant_covariates.columns.tolist()}")

        # Still remove it to match R behavior
        relevant_covariates = relevant_covariates.drop(columns=[col_to_remove])

print(f"Final covariate matrix shape: {relevant_covariates.shape}")
print(f"Columns: {relevant_covariates.columns.tolist()}")

relevant_covariates.head(1)

# %%
# First fit on training data, then center, then impute

imputer = SimpleImputer(strategy="median")  # impute column with median value
scaler = StandardScaler(with_std=False)  # only centering, no scaling!

X_train = relevant_covariates.iloc[:n_train].copy()

# First fit imputer on training data (to get medians for each column)
imputer.fit(X_train)

# Then fit scaler on imputed training data (to get means for centering)
X_train_imputed = imputer.transform(X_train)
scaler.fit(X_train_imputed)

# Now apply both transformations to all data
cov_mat_imputed = imputer.transform(relevant_covariates)
cov_mat_preprocessed = scaler.transform(cov_mat_imputed)

X_train_preprocessed = cov_mat_preprocessed[:n_train]

# QR-decomposition
Q, R = np.linalg.qr(X_train_preprocessed)

# scale the Q and R matrix appropriately
Q_scaled = Q * np.sqrt(n_train - 1)
R_scaled = R / np.sqrt(n_train - 1)

X_test = cov_mat_preprocessed[n_train:]

# %%
from helpers import comp_entropy

# aggregate review stats
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

# mutate count into probabilities
for col in ["ONE_STAR", "TWO_STAR", "THREE_STAR", "FOUR_STAR", "FIVE_STAR"]:
    review_stats[col] = review_stats[col] / review_stats["COUNT"]

# Add variation coeffient to review_stats
review_stats["COV"] = np.sqrt(review_stats["VAR"]) / review_stats["MEAN"]

# select covariates, that are relevant for training the benchmark models
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

# Add Closed column (opposite from is_open)
benchmark_covariates["Closed"] = 1 - benchmark_covariates["is_open"]

# merge covariates with aggregate review_stats
benchmark_covariates = benchmark_covariates.merge(
    review_stats, on="business_id", how="left"
)

# add logarithmic count
benchmark_covariates["l_COUNT"] = np.log(benchmark_covariates["COUNT"])

# Convert to categorical again
benchmark_covariates["category"] = benchmark_covariates["category"].astype("category")

# Convert Closed to categorical with proper labels (Closed = 1, Open = 0)
benchmark_covariates["Closed"] = (
    benchmark_covariates["Closed"].map({1: "Closed", 0: "Open"}).astype("category")
)

print(f"Benchmark covariates prepared with shape: {benchmark_covariates.shape}")

benchmark_covariates

# %%
from helpers import ModelData
from constants import PROCESSED_DATA_FOLDER

# Create ModelData instance with all data in one place
model_data = ModelData(
    n_states=None,  # not used yet, reserved for HMM models (prepare_stan_data function)
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

# save to pickle
output_path = PROCESSED_DATA_FOLDER / f"processed_data_{SEED}.pkl"
model_data.to_pickle(output_path)

print(f"Data saved to {output_path}")
print(model_data.summary())
