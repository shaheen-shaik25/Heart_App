"""Train, compare and persist heart-disease models.

Run:  python train.py

Improvements over the original notebook
  * removes duplicate rows (the raw CSV has ~720 duplicates which leak between train/test)
  * one sklearn Pipeline per model (preprocessing + estimator) -> no train/test leakage
  * stratified 5-fold cross-validation + hyper-parameter search
  * 7 models incl. Gradient Boosting, KNN and a soft-voting ensemble
  * probability calibration-friendly metrics: ROC-AUC, precision, recall, F1
  * permutation feature importance, ROC curves, confusion matrices saved to JSON
"""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import INVERT_TARGET, CATEGORICAL, FEATURES, MODEL_DIR, NUMERIC, TARGET, load_data

warnings.filterwarnings("ignore")
SEED = 42


def preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def make(est):
    return Pipeline([("prep", preprocessor()), ("clf", est)])


def candidates():
    """(name, pipeline, param_grid)"""
    return {
        "Logistic Regression": (make(LogisticRegression(max_iter=2000)),
                                {"clf__C": [0.03, 0.1, 0.3, 1, 3, 10]}),
        "SVM": (make(SVC(probability=True, random_state=SEED)),
                {"clf__C": [0.3, 1, 3, 10], "clf__gamma": ["scale", 0.01, 0.03]}),
        "Random Forest": (make(RandomForestClassifier(random_state=SEED, n_jobs=1)),
                          {"clf__n_estimators": [200, 400], "clf__max_depth": [4, 6, None],
                           "clf__min_samples_leaf": [1, 3]}),
        "Decision Tree": (make(DecisionTreeClassifier(random_state=SEED)),
                          {"clf__max_depth": [3, 4, 5, 7], "clf__min_samples_leaf": [2, 5, 10]}),
        "Gradient Boosting": (make(GradientBoostingClassifier(random_state=SEED)),
                              {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.03, 0.1],
                               "clf__max_depth": [2, 3]}),
        "KNN": (make(KNeighborsClassifier()),
                {"clf__n_neighbors": [5, 9, 13, 17], "clf__weights": ["uniform", "distance"]}),
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    raw = load_data(dedupe=False)
    df = load_data()
    print(f"Rows: {len(raw)} -> {len(df)} after removing duplicates")

    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    fitted, results, roc_curves = {}, {}, {}
    for name, (pipe, grid) in candidates().items():
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr)
        fitted[name] = gs.best_estimator_
        print(f"{name:20s} best CV AUC={gs.best_score_:.3f}  {gs.best_params_}")

    # soft-voting ensemble of the strongest members
    ens = VotingClassifier(
        [(n.replace(" ", "_"), fitted[n]) for n in ("Logistic Regression", "SVM", "Random Forest", "Gradient Boosting")],
        voting="soft")
    ens.fit(X_tr, y_tr)
    fitted["Ensemble (Voting)"] = ens

    for name, model in fitted.items():
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        cv_auc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
        fpr, tpr, _ = roc_curve(y_te, proba)
        idx = np.linspace(0, len(fpr) - 1, min(len(fpr), 40)).astype(int)
        roc_curves[name] = {"fpr": fpr[idx].round(4).tolist(), "tpr": tpr[idx].round(4).tolist()}
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
        results[name] = {
            "accuracy": accuracy_score(y_te, pred), "precision": precision_score(y_te, pred),
            "recall": recall_score(y_te, pred), "f1": f1_score(y_te, pred),
            "roc_auc": roc_auc_score(y_te, proba),
            "cv_auc_mean": cv_auc.mean(), "cv_auc_std": cv_auc.std(),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }
        results[name] = {k: (round(float(v), 4) if not isinstance(v, dict) else v) for k, v in results[name].items()}

    best = max(results, key=lambda n: (results[n]["cv_auc_mean"], results[n]["roc_auc"]))
    print("Best model:", best)

    # permutation importance on the best model
    pi = permutation_importance(fitted[best], X_te, y_te, scoring="roc_auc", n_repeats=30,
                                random_state=SEED, n_jobs=1)
    importance = sorted(({"feature": f, "importance": round(float(m), 4), "std": round(float(s), 4)}
                         for f, m, s in zip(FEATURES, pi.importances_mean, pi.importances_std)),
                        key=lambda d: -d["importance"])

    # refit every model on the full data for deployment
    for name, model in fitted.items():
        model.fit(X, y)
        joblib.dump(model, MODEL_DIR / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib")

    meta = {
        "best_model": best, "results": results, "roc": roc_curves, "importance": importance,
        "invert_target": INVERT_TARGET, "n_rows_raw": int(len(raw)), "n_rows_clean": int(len(df)),
        "class_balance": {"healthy": int((y == 0).sum()), "disease": int((y == 1).sum())},
        "medians": {c: float(df[c].median()) for c in NUMERIC},
        "modes": {c: int(df[c].mode()[0]) for c in CATEGORICAL},
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("Saved models to", MODEL_DIR)


if __name__ == "__main__":
    main()
