import os

import numpy as np
import pandas as pd

def read_pandas(fpath):
    X = pd.read_pickle(fpath)
    return X

def read_numpy(fpath):
    y = np.load(fpath)
    return y
