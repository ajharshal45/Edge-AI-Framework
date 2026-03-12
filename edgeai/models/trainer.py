"""
EdgeAI Trainer
==============
Trains lightweight scikit-learn models on domain datasets and
saves them as .pkl files into ``edgeai/models/pretrained/``.

This script runs **once** on a capable machine (laptop / desktop).
After training, edge devices only load the pretrained models and
run inference — they never retrain.

Each .pkl file stores a dictionary::

    {
        "model":         fitted estimator,
        "scaler":        fitted StandardScaler,
        "domain":        str,
        "model_name":    str,
        "accuracy":      float,
        "feature_count": int,
        "trained_at":    ISO-8601 timestamp,
    }
"""

import os
import pickle
from datetime import datetime, timezone

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from edgeai.core.preprocessor import load_training_data


# ──────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────

_DOMAINS = ["healthcare", "smartcity", "environment"]
_MODELS = ["logistic", "random_forest", "neural_network"]

_PRETRAINED_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pretrained"
)


# ──────────────────────────────────────────────────────────
#  Model factory
# ──────────────────────────────────────────────────────────

def get_model_instance(model_name):
    """
    Return a fresh, configured scikit-learn estimator.

    Args:
        model_name (str): One of ``'logistic'``, ``'random_forest'``,
                          ``'neural_network'``.

    Returns:
        sklearn estimator: Unfitted model instance.

    Raises:
        ValueError: If *model_name* is not recognised.
    """
    if model_name == "logistic":
        return LogisticRegression(
            solver="lbfgs", max_iter=200, random_state=42
        )
    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=42
        )
    elif model_name == "neural_network":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=200, random_state=42
        )
    else:
        valid = ", ".join(_MODELS)
        raise ValueError(
            f"Unknown model '{model_name}'. Valid models: {valid}"
        )


# ──────────────────────────────────────────────────────────
#  Single-model training
# ──────────────────────────────────────────────────────────

def train_single_model(domain, model_name, data_dir="data/"):
    """
    Train one model on one domain and save as a .pkl file.

    Pipeline:
        1. Load CSV via ``load_training_data``.
        2. 80/20 train/test split (stratified when possible).
        3. Fit ``StandardScaler`` on X_train only.
        4. Train model on scaled X_train.
        5. Evaluate accuracy on scaled X_test.
        6. Pickle ``{model, scaler, metadata}`` dict.

    Args:
        domain (str):     Domain name.
        model_name (str): Model type.
        data_dir (str):   Path to the data directory containing CSVs.

    Returns:
        dict: The saved payload (model, scaler, metadata).
    """
    print(f"  Training {model_name} on {domain}...")

    # 1. Load data
    X, y = load_training_data(domain, data_dir)

    # 2. Split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # stratify may fail with very small classes
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # 3. Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Train
    model = get_model_instance(model_name)
    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 6. Save
    payload = {
        "model": model,
        "scaler": scaler,
        "domain": domain,
        "model_name": model_name,
        "accuracy": round(accuracy, 4),
        "feature_count": X.shape[1],
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(_PRETRAINED_DIR, exist_ok=True)
    pkl_path = os.path.join(_PRETRAINED_DIR, f"{domain}_{model_name}.pkl")

    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)

    size_kb = os.path.getsize(pkl_path) / 1024
    print(
        f"  ✓ Saved {pkl_path} "
        f"(acc: {accuracy:.1%}, size: {size_kb:.1f} KB)"
    )

    return payload


# ──────────────────────────────────────────────────────────
#  Train everything
# ──────────────────────────────────────────────────────────

def train_all_models(data_dir="data/"):
    """
    Train all 3 model types across all 3 domains (9 total).

    Prints a summary table at the end.

    Args:
        data_dir (str): Path to the data directory.

    Returns:
        list[dict]: List of payload dicts for every trained model.
    """
    results = []

    print()
    print("═" * 50)
    print("  EdgeAI Trainer — Training all models")
    print("═" * 50)

    for domain in _DOMAINS:
        print(f"\n▶ Domain: {domain}")
        for model_name in _MODELS:
            try:
                payload = train_single_model(domain, model_name, data_dir)
                results.append(payload)
            except Exception as exc:
                print(f"  ✗ FAILED {domain}/{model_name}: {exc}")
                results.append({
                    "domain": domain,
                    "model_name": model_name,
                    "accuracy": 0.0,
                })

    # Summary table
    print()
    print("  Training Summary")
    print("  " + "═" * 46)
    print(f"  {'Domain':<14}{'Model':<18}{'Accuracy':>10}")
    print("  " + "─" * 46)
    for r in results:
        acc = r.get("accuracy", 0.0)
        print(f"  {r['domain']:<14}{r['model_name']:<18}{acc:>9.1%}")
    print("  " + "═" * 46)
    print()

    return results


# ──────────────────────────────────────────────────────────
#  Verify pretrained models
# ──────────────────────────────────────────────────────────

def verify_pretrained_models():
    """
    Check that all 9 pretrained .pkl files exist.

    Returns:
        bool: True if every expected file is present.
    """
    print("\n  Verifying pretrained models...")
    print("  " + "─" * 46)

    all_ok = True
    for domain in _DOMAINS:
        for model_name in _MODELS:
            filename = f"{domain}_{model_name}.pkl"
            path = os.path.join(_PRETRAINED_DIR, filename)
            if os.path.isfile(path):
                size_kb = os.path.getsize(path) / 1024
                print(f"  ✓ {filename:<40} {size_kb:>6.1f} KB")
            else:
                print(f"  ✗ {filename:<40} MISSING")
                all_ok = False

    print("  " + "─" * 46)
    status = "ALL PRESENT" if all_ok else "SOME MISSING"
    print(f"  Status: {status}")
    print()

    return all_ok


# ──────────────────────────────────────────────────────────
#  Standalone execution
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_all_models()
    verify_pretrained_models()
