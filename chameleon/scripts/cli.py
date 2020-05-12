import os
from typing import List

import click
import numpy as np
import pandas as pd
from scipy import io


from chameleon import errors
import chameleon.write_files as write
import chameleon.read_files as read 
from chameleon import fileio

# Legacy compatability (while migrating from old to new codebase)
from chameleon.legacy import featureselection


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> int:
    return 0


@cli.command()
@click.option("-d", "--data", 
            type=click.Path(exists=True),
            required=True, 
            help="The folder containing the prepared data e.g. kfold_myproblem")
# @click.option("--excludefile", type=str,
#                 multiple=True,
#                 help="The name of a file in the --data dir to exclude")
@click.option("-a", "--algorithm", 
            type=click.Choice(["fischer", "reliefF", "random-forest", "SVM-RFE", "simple_MI", "iterative_MI"]),
            default=["fischer", "reliefF", "random-forest", "SVM-RFE", "simple_MI", "iterative_MI"],
            multiple=True,
            help="Name of feature selection algorithms to add to the config")
@click.option("-n", "--name", 
                type=str, 
                default="pipe",
                help="Optional name for a custom pipeline config suite to add to")
@click.pass_context
def features(ctx: click.Context,
           data: str,
           algorithm: List[str],
           name: str
           ) -> None:
    """Add feature selection algorithms to the problem config"""
    catching_f = errors.catch_and_exit(features_entrypoint)
    catching_f(data, algorithm, name)


def features_entrypoint(data: str,
                      algorithm: List[str],
                      name: str
                      ) -> None:
    outdir = os.path.join(data, "configs")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    config_file = os.path.join(outdir, f"{name}.json")  
    config = {}
    if os.path.exists(config_file):
        config = read.read_json(config_file)

    datafiles = fileio.datafiles(data)
    dfile_config_template = {
        "feature_algorithms": [],
        "classifiers": []
    }

    for dfile in datafiles:
        dfile_config = config.get(dfile, dfile_config_template)
        for alg in algorithm:
            if alg not in dfile_config['feature_algorithms']:
                dfile_config['feature_algorithms'].append(alg)
        config[dfile] = dfile_config
    # import pprint as pp
    # pp.pprint(config)

    write.write_json(config, config_file)
    


@cli.command()
@click.option("-c", "--configfile", 
            type=click.Path(exists=True),
            required=True, 
            help="The relative path to the config file")
@click.option("-d", "--data", 
            type=click.Path(exists=True),
            required=True, 
            help="The folder containing the prepared data e.g. kfold_myproblem")
# @click.option("--excludefile", type=str,
#                 multiple=True,
#                 help="The name of a file in the --data dir to exclude")
@click.option("--method", 
            type=click.Choice(["normal", "multithread", "slurm"]),
            default="normal",
            help="Select the method for running the program. slurm submits the suite as a collection of jobs via Slurm, multithread sets multithreading on")
@click.pass_context
def run(ctx: click.Context,
            configfile: str,
            data: str,
            method: str
            ) -> None:
    """Run the config suite"""
    catching_f = errors.catch_and_exit(run_entrypoint)
    catching_f(configfile, data, method)


def run_entrypoint(configfile: str,
                    data: str,
                    method: str
                    ) -> None:
    config = read.read_json(configfile)
    if method == 'normal':
        for file in config.keys():
            for alg in config[file]['feature_algorithms']:
                fpath = os.path.join(data, file)
                feature_idx = featureselection.run(fpath, alg)
                print(feature_idx)
    


