import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")


PHASES = {
    "Phase 1": {
        "train_file": "BT2024025_train_var1.csv",
        "test_file": "BT2024025_test_var1.csv",
        "max_degree": 10
    },

    "Phase 2": {
        "train_file": "BT2024025_train_var2.csv",
        "test_file": "BT2024025_test_var2.csv",
        "max_degree": 20
    }
}

TARGET = "y"

# Keep this reasonably small for fast execution.
# Every alpha will still be tested for EVERY degree
ALPHAS = [
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0
]

N_SPLITS = 5
RANDOM_STATE = 42


def evaluate_model(X_poly, y, degree, alpha, model_type, kfold):
    """
    Performs 5-fold CV.

    For every fold:
        Training fold
            -> StandardScaler
            -> L1/L2 regularization
            -> Validation fold
            -> Predictions

    Returns average CV MSE, RMSE and R^2.
    """

    mse_scores = []
    r2_scores = []

    for train_idx, val_idx in kfold.split(X_poly):

        X_train_fold = X_poly[train_idx]
        X_val_fold = X_poly[val_idx]

        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]


        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)


        if model_type == "L1":
            model = Lasso(
                alpha=alpha,
                max_iter=5000,
                tol=1e-4
            )

        else:
            model = Ridge(
                alpha=alpha
            )

        model.fit(X_train_scaled, y_train_fold)

        y_pred = model.predict(X_val_scaled)

        mse = mean_squared_error(y_val_fold, y_pred)
        r2 = r2_score(y_val_fold, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)

    mean_mse = np.mean(mse_scores)
    mean_rmse = np.sqrt(mean_mse)
    mean_r2 = np.mean(r2_scores)

    return mean_mse, mean_rmse, mean_r2


def run_phase(phase_name, config):

    print("\n")
    print(f"{phase_name}")
    print("\n")

    train_file = config["train_file"]
    test_file = config["test_file"]
    max_degree = config["max_degree"]


    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    if TARGET not in train_df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found in {train_file}"
        )

    X_train = train_df.drop(columns=[TARGET]).values
    y_train = train_df[TARGET].values

    X_test = test_df.values

    print(f"Train samples : {X_train.shape[0]}")
    print(f"Features      : {X_train.shape[1]}")
    print(f"Test samples  : {X_test.shape[0]}")
    print(f"Degrees       : 1 - {max_degree}")
    print(f"Alphas        : {ALPHAS}")
    print(f"CV folds      : {N_SPLITS}")


    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = []


    best_rmse = np.inf
    best_model_type = None
    best_degree = None
    best_alpha = None

    # Flow of operations:
    # Train Data
    #     ↓
    # PolynomialFeatures
    #     ↓
    # StandardScaler
    #     ↓
    # L1/L2
    #     ↓
    # 5-Fold CV

    for degree in range(1, max_degree + 1):

        poly = PolynomialFeatures(
            degree=degree,
            include_bias=False
        )

        X_poly = poly.fit_transform(X_train)

        for model_type in ["L1", "L2"]:

            for alpha in ALPHAS:

                mse, rmse, r2 = evaluate_model(
                    X_poly=X_poly,
                    y=y_train,
                    degree=degree,
                    alpha=alpha,
                    model_type=model_type,
                    kfold=kfold
                )

                results.append({
                    "Regularization": model_type,
                    "Degree": degree,
                    "Alpha": alpha,
                    "MSE": mse,
                    "RMSE": rmse,
                    "R2": r2
                })

                print(
                    f"{model_type:>2} | "
                    f"Degree = {degree:2d} | "
                    f"Alpha = {alpha:<8g} | "
                    f"MSE = {mse:.6f} | "
                    f"RMSE = {rmse:.6f} | "
                    f"R² = {r2:.6f}"
                )


                if rmse < best_rmse:

                    best_rmse = rmse
                    best_model_type = model_type
                    best_degree = degree
                    best_alpha = alpha

    # COMPARE CV RMSE AND SELECT BEST

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="RMSE",
        ascending=True
    )

    print("\n")
    print("TOP 10 MODELS ACCORDING TO 5-FOLD CV RMSE")
    print("\n")

    print(
        results_df.head(10).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print("\n")
    print("BEST MODEL")
    print("\n")

    print(f"Regularization : {best_model_type}")
    print(f"Degree         : {best_degree}")
    print(f"Alpha          : {best_alpha}")
    print(f"CV MSE         : {results_df.iloc[0]['MSE']:.6f}")
    print(f"CV RMSE        : {results_df.iloc[0]['RMSE']:.6f}")
    print(f"CV R²          : {results_df.iloc[0]['R2']:.6f}")




    best_poly = PolynomialFeatures(
        degree=best_degree,
        include_bias=False
    )

    X_train_poly = best_poly.fit_transform(X_train)
    X_test_poly = best_poly.transform(X_test)

    best_scaler = StandardScaler()

    X_train_scaled = best_scaler.fit_transform(X_train_poly)
    X_test_scaled = best_scaler.transform(X_test_poly)

    if best_model_type == "L1":

        best_model = Lasso(
            alpha=best_alpha,
            max_iter=5000,
            tol=1e-4
        )

    else:

        best_model = Ridge(
            alpha=best_alpha
        )

    
    best_model.fit(X_train_scaled, y_train)


    predictions = best_model.predict(X_test_scaled)

    prediction_df = pd.DataFrame({
        "prediction": predictions
    })

    phase_num = phase_name.split()[-1]
    output_file = f"BT2024025_pred_var{phase_num}.csv"

    prediction_df.to_csv(
        output_file,
        index=False
    )

    results_file = (
        phase_name.lower()
        .replace(" ", "_")
        + "_cv_results.csv"
    )

    results_df.to_csv(
        results_file,
        index=False
    )

    return {
        "best_model": best_model,
        "best_poly": best_poly,
        "best_scaler": best_scaler,
        "best_degree": best_degree,
        "best_alpha": best_alpha,
        "best_model_type": best_model_type,
        "results": results_df,
        "predictions": prediction_df
    }

def plot_mse_vs_degree(csv_file, title, output_file):

    df = pd.read_csv(csv_file)

    best_per_degree = (
        df.loc[df.groupby("Degree")["MSE"].idxmin()]
        .sort_values("Degree")
    )

    best = best_per_degree.loc[
        best_per_degree["MSE"].idxmin()
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        best_per_degree["Degree"],
        best_per_degree["MSE"],
        marker="o",
        linewidth=2
    )

    # Highlight best degree
    plt.scatter(
        best["Degree"],
        best["MSE"],
        s=100,
        zorder=5
    )

    plt.annotate(
        f"Best: Degree {int(best['Degree'])}\n"
        f"MSE = {best['MSE']:.6f}",
        xy=(best["Degree"], best["MSE"]),
        xytext=(10, 15),
        textcoords="offset points"
    )

    plt.xlabel("Polynomial Degree")
    plt.ylabel("Cross-Validation MSE")
    plt.title(title)

    plt.xticks(
        range(
            int(best_per_degree["Degree"].min()),
            int(best_per_degree["Degree"].max()) + 1
        )
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

if __name__ == "__main__":

    print("\n")
    print("# POLYNOMIAL REGRESSION - BOTH PHASES")

    all_results = {}

    for phase_name, config in PHASES.items():

        if not os.path.exists(config["train_file"]):
            print(
                f"\nERROR: {config['train_file']} not found."
            )
            continue

        if not os.path.exists(config["test_file"]):
            print(
                f"\nERROR: {config['test_file']} not found."
            )
            continue

        all_results[phase_name] = run_phase(
            phase_name,
            config
        )

    # PLOT MSE VS DEGREE
    plot_mse_vs_degree(
        "phase_1_cv_results.csv",
        "Phase 1: Cross-Validation MSE vs Polynomial Degree",
        "phase_1_mse_vs_degree.png"
    )

    plot_mse_vs_degree(
        "phase_2_cv_results.csv",
        "Phase 2: Cross-Validation MSE vs Polynomial Degree",
        "phase_2_mse_vs_degree.png"
    )