import os

import click
import numpy as np
import pandas as pd
from scipy import io
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from chameleon import errors
import chameleon.write_files as write
import chameleon.read_files as read



@click.group()
@click.pass_context
def cli(ctx: click.Context) -> int:
    return 0


@cli.command()
@click.option("--ftype", type=click.Choice(["mat", "arff"]),
              required=True, help="Type of data file")
@click.option("--fpath", type=click.Path(exists=True),
              help="Path to file")
@click.option("--name", type=str, required=True,
              help="Name of output files e.g. '--name foo' will make files foo_X.pkl and foo_y.npy")
@click.pass_context
def format(ctx: click.Context,
           ftype: str,
           fpath: str,
           name: str
           ) -> None:
    """Build a covariate file and target file from the input file."""
    catching_f = errors.catch_and_exit(format_entrypoint)
    catching_f(ftype, fpath, name)


def format_entrypoint(ftype: str,
                      fpath: str,
                      name: str
                      ) -> None:
    outdir = os.path.join(os.getcwd(), f"proc_data")
    out_X_name = f"X_{name}.pkl"
    out_y_name = f"y_{name}.npy"


    X, y = None, None
    if ftype == 'mat':
        in_file = io.loadmat(fpath)
        X = pd.DataFrame(in_file['X'], dtype=float)
        y = pd.DataFrame(in_file['Y'])
    elif ftype == 'arff':
        data, metadata = io.arff.loadarff(fpath)
        df = pd.DataFrame(data)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    # convert classes from whatever datatype they are to binary integers (0 and 1)
    y_values = np.unique(y)
    if len(y_values) > 2:
        raise errors.NonBinaryTargets()
    y_binary = np.array(y == y_values[0], dtype=int)
    
    write.write_pandas(X, outdir, out_X_name)
    write.write_numpy(y, outdir, out_y_name)



@cli.command()
@click.option("--xfile", type=click.Path(exists=True), required=True,
              help="The name of the X data. e.g. '--Xfile proc_data/foo_X.pkl'")
@click.option("--yfile", type=click.Path(exists=True), required=True,
              help="The name of the y data. e.g. '--yfile proc_data/foo_y.npy'")
@click.option("--k", type=int, default=5,
              help="The number of folds to partition the data into"
              "fold (counting from 1), second is total folds.")
@click.option("--random_seed", type=int, default=666,
              help="Random state for assigning data to folds")
@click.option("--name", type=str, required=True,
              help="Name of the output folder e.g. myproblem will create kfold_myproblem")
@click.option("--normalise/--no-normalise", is_flag=True, default=True,
              help="Normalise the the data ")
@click.option("--stratified/--not-stratified", is_flag=True, default=True,
              help="Apply stratified kfold")
@click.pass_context
def kfold(ctx: click.Context,
          xfile: str,
          yfile: str,
          k: int,
          random_seed: int,
          name: str,
          normalise: bool,
          stratified: bool
          ) -> None:
    """Build a covariate file and target file from the input file."""
    catching_f = errors.catch_and_exit(kfold_entrypoint)
    catching_f(xfile, yfile, k, random_seed, name, normalise, stratified)


def kfold_entrypoint(xfile: str,
                     yfile: str,
                     k: int,
                     random_seed: int,
                     name: str,
                     normalise: bool,
                     stratified: bool
                     ) -> None:
    outdir_relative = f"kfold_{name}"
    outdir = os.path.join(os.getcwd(), outdir_relative)

    X = read.read_pandas(xfile)
    y = read.read_numpy(yfile)
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_seed)
    pid = 0
    for train_idx, test_idx in kfold.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        if normalise:
            scalar =  StandardScaler().fit(X_train)
            X_train = scalar.transform(X_train)
            X_test = scalar.transform(X_test)
        fold = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y[train_idx],
            'y_test': y[test_idx]
        }
        pid += 1
        
        outfile = f"fold{pid}of{k}.pkl"
        write.write_dictionary(fold, outdir, outfile)
