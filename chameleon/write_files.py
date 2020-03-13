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

def write_feature_pipe(files, selector_list, outdir, fname):
    outfile = open(os.path.join(outdir, fname), "w")
    file_list = " ".join(files)
    algorithm_list = " ".join(selector_list)
    outfile.write(f"declare -a fileList=( {file_list} )")
    outfile.write(f"declare -a algorithmList=( {algorithm_list} )")

    outfile.write('\n')
    outfile.write('for alg in "${algorithmList[@]}"; do')
    outfile.write('    for file in "${file_list[@]}"; do')
    outfile.write('        echo "running $alg algorithm for $file"')
    outfile.write('        $FS_PROJECT_PATH/src/bash/run-feature.sh $alg fold')
    # outfile.write()

