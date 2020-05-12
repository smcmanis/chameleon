import os
import json

import numpy as np
import pandas as pd

def read_pandas(fpath):
    X = pd.read_pickle(fpath)
    return X

def read_numpy(fpath):
    y = np.load(fpath)
    return y

def read_json(fpath):
    with open(fpath) as json_file:
        data = json.load(json_file)
    return data