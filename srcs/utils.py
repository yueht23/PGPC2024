from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_poisson_deviance, max_error


def load_mtpl2(n_samples=None):
    """Fetch the French Motor Third-Party Liability Claims dataset.

    Parameters
    ----------
    n_samples: int, default=None
      number of samples to select (for faster run time). Full dataset has
      678013 samples.
    """
    # freMTPL2freq dataset from https://www.openml.org/d/41214
    df_freq = fetch_openml(data_id=41214, as_frame=True).data
    df_freq["IDpol"] = df_freq["IDpol"].astype(int)
    df_freq.set_index("IDpol", inplace=True)

    # freMTPL2sev dataset from https://www.openml.org/d/41215
    df_sev = fetch_openml(data_id=41215, as_frame=True).data

    # sum ClaimAmount over identical IDs
    df_sev = df_sev.groupby("IDpol").sum()

    df = df_freq.join(df_sev, how="left")
    df["ClaimAmount"].fillna(0, inplace=True)

    # unquote string fields
    for column_name in df.columns[df.dtypes.values == object]:
        df[column_name] = df[column_name].str.strip("'")
    return df.iloc[:n_samples]


def load_data(n_samples=None):
    try:
        # load data from local pickle
        df = pd.read_pickle("mtpl2.pkl")
    except FileNotFoundError:
        # or download from openml
        df = load_mtpl2()
        df.to_pickle("mtpl2.pkl")
    return df.iloc[:n_samples]


def print_scores(scores):
    for key, value in scores.items():
        if key in ["model"]:
            print(f"{key}: {value}")
        else:
            print("{s}:{v:.4f};(std:{t:.4f})".format(s=key, v=np.mean(value), t=np.std(value)))


def get_scores(model_name):
    return {
        # please specify the model name here
        "model": model_name,

        # Mean Absolute Error for training set and testing set
        "train_MAE": [],
        "test_MAE": [],

        # Max Error for training set and testing set
        "train_Max_Error": [],
        "test_Max_Error": [],

        # Mean Poisson Deviance for training set and testing set
        "train_Mean_Poisson_Deviance": [],
        "test_Mean_Poisson_Deviance": [],

        # Explained Variance Score for training set and testing set
        "train_explained_variance": [],
        "test_explained_variance": [],

        # Time complexity for training set and testing set
        "train_time": [],
        "test_time": [],

        # Space complexity for training set and testing set
        "train_memory": [],
        "test_memory": []
    }


def calculate_metrics(scores, y_train, y_pred_train, y_test, y_pred_test):
    # Mean Absolute Error for training set and testing set
    scores["train_MAE"].append(mean_squared_error(y_train, y_pred_train))
    scores["test_MAE"].append(mean_squared_error(y_test, y_pred_test))

    # Max Error for training set and testing set
    scores["train_Max_Error"].append(max_error(y_train, y_pred_train))
    scores["test_Max_Error"].append(max_error(y_test, y_pred_test))

    # Mean Poisson Deviance for training set and testing set
    scores["train_Mean_Poisson_Deviance"].append(mean_poisson_deviance(y_train, y_pred_train))
    scores["test_Mean_Poisson_Deviance"].append(mean_poisson_deviance(y_test, y_pred_test))

    # Explained Variance Score for training set and testing set
    scores["train_explained_variance"].append(explained_variance_score(y_train, y_pred_train))
    scores["test_explained_variance"].append(explained_variance_score(y_test, y_pred_test))
    return scores
