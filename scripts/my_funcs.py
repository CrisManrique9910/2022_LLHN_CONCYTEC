import numpy as np
import pandas as pd

def my_arctan(y,x):

    arctan = np.arctan2(y,x)
    if not isinstance(x, (pd.Series, np.ndarray)):
        arctan = np.asarray(arctan)
    arctan[arctan < 0] += 2*np.pi
    return arctan