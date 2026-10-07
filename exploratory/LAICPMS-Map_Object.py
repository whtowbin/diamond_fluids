# %%
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Union
import matplotlib.pyplot as plt
from copy import deepcopy

# %%


@dataclass(order=True)
class LAICPMS_Map:
    X: np.ndarray = None  # Sequential array; spatial X
    Y: np.ndarray = None  # Sequential array; spatial Y
    Z: np.ndarray = None  #
    X_Unit: str = None
    Y_Unit: str = None
    metadata: Dict = None
    kwargs: Dict = None  # defines kwargs, should probably have a metadata dict
    baseline = None
