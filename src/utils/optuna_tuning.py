"""Optuna search with time-series CV. No shuffle — that would leak the future."""
import numpy as np
import optuna
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 20


def _cv_rmse(model, X, y, tscv) -> float:
    """Mean RMSE on the time-series folds."""
    scores = []
    for train_idx, val_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        scores.append(np.sqrt(mean_squared_error(y.iloc[val_idx], preds)))
    return float(np.mean(scores))


def tune_ridge(X_train, y_train, tscv, n_trials: int = N_TRIALS):
    """Ridge with scaling, so big and small features are treated fairly."""
    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.01, 100.0, log=True)
        pipeline = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        return _cv_rmse(pipeline, X_train, y_train, tscv)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_pipeline = make_pipeline(StandardScaler(), Ridge(**study.best_params))
    return best_pipeline.fit(X_train, y_train)


def tune_random_forest(X_train, y_train, tscv, n_trials: int = N_TRIALS):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [8, 12, 16, 24, None]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
        return _cv_rmse(RandomForestRegressor(random_state=42, **params),
                         X_train, y_train, tscv)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return RandomForestRegressor(random_state=42, **study.best_params).fit(X_train, y_train)


def tune_xgboost(X_train, y_train, tscv, n_trials: int = N_TRIALS):
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        return _cv_rmse(xgb.XGBRegressor(random_state=42, **params),
                         X_train, y_train, tscv)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return xgb.XGBRegressor(random_state=42, **study.best_params).fit(X_train, y_train)


def tune_lightgbm(X_train, y_train, tscv, n_trials: int = N_TRIALS):
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
        return _cv_rmse(lgb.LGBMRegressor(random_state=42, verbosity=-1, **params),
                         X_train, y_train, tscv)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return lgb.LGBMRegressor(random_state=42, verbosity=-1, **study.best_params).fit(X_train, y_train)
