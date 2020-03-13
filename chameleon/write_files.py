import fileinput
import os
import pickle

import numpy as np
import pandas as pd

def directory_check(dpath):
    if not os.path.exists(dpath):
        os.makedirs(dpath)

def write_pandas(df, outdir, fname):
    directory_check(outdir)
    outfile = os.path.join(outdir, fname)
    df.to_pickle(outfile)

def write_numpy(array, outdir, fname):
    directory_check(outdir)
    outfile = os.path.join(outdir, fname)
    np.save(outfile, array)

def write_dictionary(dic, outdir, fname):
    directory_check(outdir)
    outfile = os.path.join(outdir, fname)
    pickle.dump(dic, open(outfile, 'wb'))


