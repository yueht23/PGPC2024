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

    # set datatype
    df["VehAge"] = df["VehAge"].astype("Int64")
    df["DrivAge"] = df["DrivAge"].astype("Int64")

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
    tmp = []
    model_name = scores["model"]
    for key, value in scores.items():
        if key in ["model"]:
            continue
        else:
            tmp.append({
                "Model": model_name,
                "Type": key.split("_")[0],
                "Metric": key.split("_")[1],
                "Mean": np.mean(value),
                "Std": np.std(value)
            })

    return pd.DataFrame(tmp).sort_values(by=["Type", "Metric"])


def get_scores(model_name):
    return {
        # please specify the model name here
        "model": model_name,

        # Mean Absolute Error for training set and testing set
        "train_MAE": [],
        "test_MAE": [],

        # Max Error for training set and testing set
        "train_MaxError": [],
        "test_MaxError": [],

        # Mean Poisson Deviance for training set and testing set
        "train_MeanPoissonDeviance": [],
        "test_MeanPoissonDeviance": [],

        # Proportion of Deviance Explained (PDE) for training set and testing set
        "train_PDE": [],
        "test_PDE": [],

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
    scores["train_MaxError"].append(max_error(y_train, y_pred_train))
    scores["test_MaxError"].append(max_error(y_test, y_pred_test))

    # Mean Poisson Deviance for training set and testing set
    scores["train_MeanPoissonDeviance"].append(mean_poisson_deviance(y_train, y_pred_train))
    scores["test_MeanPoissonDeviance"].append(mean_poisson_deviance(y_test, y_pred_test))

    # Explained Variance Score for training set and testing set
    scores["train_PDE"].append(explained_variance_score(y_train, y_pred_train))
    scores["test_PDE"].append(explained_variance_score(y_test, y_pred_test))
    return scores


if __name__ == "__main__":
    scores = get_scores("model_name")
    scores = calculate_metrics(scores, np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3]),
                               np.array([1, 2, 3]))
    scores["test_time"] = [1, 2, 3]
    scores["train_time"] = [1, 2, 3]
    scores["test_memory"] = [1, 2, 3]
    scores["train_memory"] = [1, 2, 3]

    print(print_scores(scores))
