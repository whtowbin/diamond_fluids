"""Load the Weiss et al. HDF compilation and build trace-element log-ratio features.

Shared by the Fluid_Comp_Analysis scripts and make_figures.py so they all build
features and the classifier the same way.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PATH = "Weiss_et_al-RiMG-Global_Diamond_HDF_compilation_Aug_2021.xlsx"
SHEET = "HDF_Limited_Col"
LABEL = "HDF Type (calculated)"
HDF_TYPES = ["saline", "silicic", "silicic - low-Mg carbonatitic", "low-Mg carbonatitic", "high-Mg carbonatitic"]

# Elements present in most of the compilation and measurable by LA-ICP-MS in twinning wisps.
# Adding Th, REE, Cs, K, Ti, Hf, Ta or Mg, Al, Na did not improve the held-out-paper score (10/7/25).
ELEMENTS = ["Rb", "Sr", "Y", "Zr", "Nb", "Ba", "La", "Pb", "U"]
REFERENCE = "Ce"


def load_compilation(path=PATH):
    return pd.read_excel(path, sheet_name=SHEET)


def concentration(df, element):
    """Concentration in ppm, with blanks, zeros and below-detection strings ("<DL", "nd") as NaN."""
    c = pd.to_numeric(df[element], errors="coerce")
    return c.where(c > 0)


def log_ratios(df, elements=ELEMENTS, reference=REFERENCE):
    """log(element / reference) for each element, computed from the concentrations.

    The compilation's own ratio columns store a missing numerator as 0 (every Pb/U == 0 row has
    no Pb measured), so they cannot be used directly. Ratios span orders of magnitude and most
    are below 1, where log(x + 1) squashes them toward 0, so the plain log is used.
    """
    ref = concentration(df, reference)
    return pd.DataFrame(
        {f"log {e}/{reference}": np.log(concentration(df, e) / ref) for e in elements},
        index=df.index,
    )


def has_trace_data(features, min_ratios=3):
    """Rows with at least `min_ratios` usable log ratios; the rest are mostly major-element-only analyses."""
    return features.notna().sum(axis=1) >= min_ratios


def make_classifier(C=0.1):
    """Median imputation, scaling and balanced logistic regression in one pipeline.

    Keeping the imputer and scaler in the pipeline means each CV fold fits them on its training
    rows only. class_weight="balanced" stops the model favouring the largest class. C=0.1 scored
    best of 0.03-1 for HDF type, though all were within ~0.02.
    """
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=5000),
    )
