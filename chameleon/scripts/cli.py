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
from chameleon.legacy import featureselection, classify


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> int:
    return 0


@cli.command()
@click.option("-d", "--data", 
            type=click.Path(exists=True),
            required=True, 
            help="The folder containing the prepared data e.g. kfold_myproblem")
@click.option("-f", "--featureselector", 
            type=click.Choice(["fischer", "reliefF", "random-forest", "SVM-RFE", "simple_MI", "iterative_MI"]),
            default=["fischer", "reliefF", "random-forest", "SVM-RFE", "simple_MI", "iterative_MI"],
            multiple=True,
            help="Name of feature selection algorithms to add to the config")
@click.option("-c", "--classifier", 
            type=click.Choice(["naive-bayes", "kNN", "logistic-regression", "neural-net", "random-forest", "SVM"]),
            default=["naive-bayes", "kNN", "logistic-regression", "neural-net", "random-forest", "SVM"],
            multiple=True,
            help="Name of classifier algorithms to add to the config")
@click.option("-n", "--name", 
            type=str, 
            default="pipe",
            help="Name of custom pipeline config")
@click.pass_context
def pipe(ctx: click.Context,
        data: str,
        featureselector: List[str],
        classifier: List[str],
        name: str
        ) -> None:
    """Create pipeline config"""
    catching_f = errors.catch_and_exit(pipe_entrypoint)
    catching_f(data, featureselector, classifier, name)


def pipe_entrypoint(data: str,
        featureselector: List[str],
        classifier: List[str],
        name: str
        ) -> None:

    features_entrypoint(data, featureselector, name)

    configpath = os.path.join(data, f"configs/{name}.json")
    classifiers_entrypoint(configpath, classifier)


@cli.command()
@click.option("-d", "--data", 
            type=click.Path(exists=True),
            required=True, 
            help="The folder containing the prepared data e.g. kfold_myproblem")
# @click.option("--excludefile", type=str,
#                 multiple=True,
#                 help="The name of a file in the --data dir to exclude")
@click.option("-f", "--featureselector", 
            type=click.Choice(["fischer", "reliefF", "random-forest", "SVM-RFE", "simple_MI", "iterative_MI"]),
            default=["fischer", "reliefF", "random-forest", "SVM-RFE", "simple_MI", "iterative_MI"],
            multiple=True,
            help="Name of feature selection algorithms to add to the config")
@click.option("-n", "--name", 
                type=str, 
                default="pipe",
                help="Optional name for a custom pipeline config suite to create or add to")
@click.pass_context
def features(ctx: click.Context,
           data: str,
           featureselector: List[str],
           name: str
           ) -> None:
    """Add feature selection algorithms to the problem config"""
    catching_f = errors.catch_and_exit(features_entrypoint)
    catching_f(data, featureselector, name)


def features_entrypoint(data: str,
                        featureselector: List[str],
                        name: str
                        ) -> None:
    outdir = os.path.join(data, "configs")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    config_file = os.path.join(outdir, f"{name}.json") 
    if os.path.exists(config_file):
        config = read.read_json(config_file)
    else:
        config = {}

    datafiles = fileio.datafiles(data)

    for dfile in datafiles:
        dfile_config = config.get(dfile, {})
        selectors = dfile_config.get("feature_algorithms", [])
        new_fs = list(set(featureselector)- set(selectors))
        dfile_config['feature_algorithms'] = selectors + new_fs
        config[dfile] = dfile_config

    write.write_json(config, config_file)
    


@cli.command()
@click.option("-p", "--configpath", 
            type=click.Path(exists=True),
            required=True, 
            help="The relative path to the config file")
@click.option("-c", "--classifier", 
            type=click.Choice(["naive-bayes", "kNN", "logistic-regression", "neural-net", "random-forest", "SVM"]),
            default=["naive-bayes", "kNN", "logistic-regression", "neural-net", "random-forest", "SVM"],
            multiple=True,
            help="Name of classifier algorithms to add to the config")
@click.pass_context
def classifiers(ctx: click.Context,
           configpath: str,
           classifier: List[str]
           ) -> None:
    """Add classifiers to the pipeline"""
    catching_f = errors.catch_and_exit(classifiers_entrypoint)
    catching_f(configpath, classifier)


def classifiers_entrypoint(configpath: str,
                        classifier: List[str]
                        ) -> None:
    config = read.read_json(configpath)
    for pipe in config.values():
        clfs = pipe.get("classifiers", [])
        new_clfs = list(set(classifier)- set(clfs))
        pipe['classifiers'] = clfs + new_clfs
    
    write.write_json(config, configpath)
    


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
            if feature_selection:
                run_feature_selection(config, file, data, feature_dir)
            if predict:
                run_predictions(config, file, data, feature_dir, predictions_dir, range)



