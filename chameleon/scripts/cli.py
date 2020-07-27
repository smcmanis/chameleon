import os
from typing import List
from pprint import pprint

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
        datafiles = fileio.datafiles(data)
        config = {'data': [f for f in datafiles]}

    # datafiles = fileio.datafiles(data)
    
    # for dfile in datafiles:
    #     dfile_config = config.get(dfile, {})
    #     selectors = dfile_config.get("feature_algorithms", [])
    #     new_fs = list(set(featureselector)- set(selectors))
    #     dfile_config['feature_algorithms'] = selectors + new_fs
    #     config[dfile] = dfile_config

    selectors = config.get("feature_selection", [])
    new_fs = list(set(featureselector) - set(selectors))
    config['feature_selection'] = selectors + new_fs

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
    
    # for pipe in config.values():
    #     clfs = pipe.get("classifiers", [])
    #     new_clfs = list(set(classifier)- set(clfs))
    #     pipe['classifiers'] = clfs + new_clfs

    classifiers = config.get("classifiers", [])
    new_clfs = list(set(classifier) - set(classifiers))
    config['classifiers'] = classifiers + new_clfs
    
    write.write_json(config, configpath)
    


@cli.command()
@click.option("-p", "--pipe", 
            type=click.Path(exists=True),
            required=True, 
            help="The relative path to the pipe config file")
@click.option("-d", "--data", 
            type=click.Path(exists=True),
            required=True, 
            help="The folder containing the prepared data e.g. kfold_myproblem")
# @click.option("--excludefile", type=str,
#                 multiple=True,
#                 help="The name of a file in the --data dir to exclude")
@click.option("-m", "--method", 
            type=click.Choice(["normal", "slurm"]),
            default="normal",
            help="Select the method for running the program. With 'normal', the pipelines will be executed iteratively, and with 'slurm', the pipelines are submitted as a collection of jobs via Slurm")
@click.option("--feature_selection", 
            type=bool,
            default=True,
            help="Run feature selection in pipeline")
@click.option("--predict", 
            type=bool,
            default=True,
            help="Run predictions in pipeline")
@click.option("-n", "--n_features", 
            type=int,
            default=50,
            help="Number of features to use in classifier predictions (the top 'n' features)")    
@click.pass_context
def run(ctx: click.Context,
            pipe: str,
            data: str,
            method: str,
            feature_selection: bool,
            predict: bool,
            n_features: int
            ) -> None:
    """Run the config suite"""
    catching_f = errors.catch_and_exit(run_entrypoint)
    catching_f(pipe, data, method, feature_selection, predict, n_features)


def run_entrypoint(pipe: str,
                    data: str,
                    method: str,
                    feature_selection: bool,
                    predict: bool,
                    n_features: tuple
                    ) -> None:
    config = read.read_json(pipe)

    feature_dir = os.path.join(data, "features")
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    # Can probably be done more efficiently
    if method == 'normal':
        results = {}
        if feature_selection: 
            run_feature_selection(config, data, feature_dir)
        if predict:
            results = run_predictions(config, data, feature_dir, n_features)
            pprint(results)
    

    
def run_feature_selection(config, data_dir, feature_dir):
    for file in config['data']:
        fpath = os.path.join(data_dir, file)
        for alg in config['feature_selection']:
            feature_idx = featureselection.run(fpath, alg)
            # Remove .pkl from fold filename
            fname = f"{file[:-4]}_{alg}.npy"
            write.write_numpy(feature_idx, feature_dir, fname)

def run_predictions(config, data_dir, feature_dir, n_features):
    results = {}
    for alg in config['feature_selection']:
        results[alg] = {}
        for clf in config['classifiers']:
            aggregate_score = 0
            for file in config['data']:
                features_fname = f"{file[:-4]}_{alg}.npy"
                features_path = os.path.join(feature_dir, features_fname) 
                feature_idx = read.read_numpy(features_path)
                data_path = os.path.join(data_dir, file)
                score = classify.run(data_path, clf, feature_idx, n_features)
                aggregate_score += score
            avg_score = aggregate_score / len(config['data'])
            results[alg][clf] = avg_score

    return results
        

