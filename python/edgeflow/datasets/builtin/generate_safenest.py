"""
Generates a synthetic but realistic SafeNest sensor dataset.

SafeNest sensors:
    - mq2_raw   : MQ-2 gas sensor ADC value (0-4095)
    - pir_state : PIR motion sensor (0 or 1)

Labels:
    - SAFE   : normal conditions
    - DANGER : gas leak detected
"""

import numpy as np
import pandas as pd
from typing import Optional


def generate_safenest_dataset(
    n_samples: int = 500,
    random_state: int = 42,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate synthetic SafeNest sensor data.

    DANGER conditions:
        mq2_raw > 2200 (with gaussian noise)
        pir_state = random (gas leak can happen anytime)

    SAFE conditions:
        mq2_raw < 1800 (with gaussian noise)
        pir_state = random (normal anytime)

    Parameters
    ----------
    n_samples : total samples (split 50/50 SAFE/DANGER)
    random_state : for reproducibility
    output_path : if provided, saves CSV to this path

    Returns
    -------
    pd.DataFrame with columns: mq2_raw, pir_state, label
    """
    rng = np.random.RandomState(random_state)

    half = n_samples // 2

    # SAFE samples
    safe_mq2 = rng.normal(loc=900, scale=200, size=half).clip(100, 1799)
    safe_pir = rng.randint(0, 2, size=half)
    safe_labels = ["SAFE"] * half

    # DANGER samples
    danger_mq2 = rng.normal(loc=2800, scale=300, size=half).clip(2201, 4095)
    danger_pir = rng.randint(0, 2, size=half)
    danger_labels = ["DANGER"] * half

    # Combine and shuffle
    mq2 = np.concatenate([safe_mq2, danger_mq2]).astype(int)
    pir = np.concatenate([safe_pir, danger_pir])
    labels = safe_labels + danger_labels

    df = pd.DataFrame({
        "mq2_raw": mq2,
        "pir_state": pir,
        "label": labels,
    })

    # Shuffle rows
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"[EdgeFlow] Dataset saved to {output_path}")
        print(f"[EdgeFlow] Samples: {len(df)} | SAFE: {half} | DANGER: {half}")

    return df


if __name__ == "__main__":
    df = generate_safenest_dataset(n_samples=500, output_path="safenest_data.csv")
    print(df.head(10))
