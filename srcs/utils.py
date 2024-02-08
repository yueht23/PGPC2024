from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score


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
    print("Model:", scores["model"])
    print("Train MAE:{:.4f},std:{:.4f}".format(np.mean(scores["train_MAE"]), np.std(scores["train_MAE"])))
    print("Test  MAE:{:.4f},std:{:.4f}".format(np.mean(scores["test_MAE"]), np.std(scores["test_MAE"])))
    print("Train Explained Variance:{:.4f},std:{:.4f}".format(np.mean(scores["train_explained_variance"]),
                                                              np.std(scores["train_explained_variance"])))
    print("Test  Explained Variance:{:.4f},std:{:.4f}".format(np.mean(scores["test_explained_variance"]),
                                                              np.std(scores["test_explained_variance"])))
    print("Train Time:{:.2f},std:{:.2f}".format(np.mean(scores["train_time"]), np.std(scores["train_time"])))
    print("Test  Time:{:.2f},std:{:.2f}".format(np.mean(scores["test_time"]), np.std(scores["test_time"])))


def get_scores(model_name):
    return {
        # please specify the model name here
        "model": model_name,

        # Mean Absolute Error for training set and testing set
        "train_MAE": [],
        "test_MAE": [],

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
    scores["train_MAE"].append(mean_squared_error(y_train, y_pred_train))
    scores["test_MAE"].append(mean_squared_error(y_test, y_pred_test))
    scores["train_explained_variance"].append(explained_variance_score(y_train, y_pred_train))
    scores["test_explained_variance"].append(explained_variance_score(y_test, y_pred_test))
    return scores