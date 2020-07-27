import os
from pprint import pprint

import click

from chameleon import errors
import chameleon.write_files as write
import chameleon.read_files as read 

# Legacy compatability (while migrating from old to new codebase)
from chameleon.legacy import featureselection, classify


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> int:
    return 0


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
        

