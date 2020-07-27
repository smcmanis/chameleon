# chameleon

\[One-sentence summary\]


## Introduction
 \[Features, background, about, etc.\]


## Installation

### Prerequisites

 The following will need to be installed before installing Chameleon:

 - Python 3.6+
 <!-- - Java
 - Heaps of other things

While not necessary, Chameleon is optimised for using job scheduled high-performance computing. This significantly reduces model training times.  -->

### Installing Chameleon

Clone the repository to your desired directory and simply run

```bash
$ pip install .
```

or, if you plan on doing development

```bash
$ pip install -e .[dev]
```


## Outline

The basic steps in using Chameleon are:

1. Import and format a dataset with `chameleon-data format`,
2. Create *k*-fold cross-validation partitions with `chameleon-data kfold`,
3. Configure the full pipeline with `chameleon-pipe pipe`. Alternatively, individually configure the pipeline using `chameleon-pipe features` to add feature selection algorithms and `chameleon-pipe classifiers` to add classifiers.
4. Run the pipeline with `chameleon run`.

<!-- ![Pipeline workflow diagram](pipelinediagram.svg) -->

Every feature selection method has been configured to return all features in a list ordered by importance in descending order. 

## Data Prerequisites

Since Chameleon is currently in the prototype stage, there are very strict requirements for data input:

1. Features and targets must be in one of the following formats:
    - `.mat` - A single file in the same format as the biological datasets seen [here](http://featureselection.asu.edu/datasets.php).
    - `.arff` - A single file with the targets as the last attribute.
2. All features must be continuous.
3. Targets must be binary.
4. No missing values.

## Usage Example

<!-- Before starting, we must make sure that the data files we want to import are located in the `/path/to/chameleon/data/external` directory.  -->

### 1. Import the data

Say we have a `.mat` file located at `/path/to/data.mat`. First, we need to extract the data from this file so it is in the correct format for chameleon. 

```bash
$ chameleon-data format --ftype mat --fpath /path/to/data.mat --name foo
```

This command will produce two files. One for the features `X_foo.pkl` and another for the targets `y_foo.npy`. They will be located in the `proc_data` folder of the current working directory. The folder will be created if it doesn't already exist.


### 2. Create cross-validation partitions

Chameleon currently uses *k*-fold cross-validation, so we are now going to randomly partition the data into *k* folds to make *k* train/test sets.

```bash
$ chameleon-data kfold --xfile proc_data/X_foo.pkl --yfile proc_data/y_foo.npy --name myproblem
```
The default *k* is 5, but can be set using `--k`. The *k*-fold method is stratified by default but can be suppressed using the `--not-stratified` flag. Similarly, the data will be normalised by default unless the `--no-normalise` flag is included.
This command will create a new folder in the current directory called `kfold_myproblem` that will contain *k* files in the format `fold1of*k*.pkl`. These files contain the unique sets of training and test data for the fold.


### 3. Configure feature selection choices
The chameleon utility provides a command line option for quickly setting up a config file for your new problem that specifies the feature selection algorithms and the classifiers that you want to run for the data files. 
The config file can be manually edited without issue.
To start, create the config file with the default feature selection algorithms using the following command: 
 
```bash
$ chameleon-pipe features -d kfold_myproblem 
```
Where `-d` specifies the directory containing your problem. This will create a pipeline configuration file that stores the data file names, feature selection algorithms, and classifiers. The config file is saved in the `kfold_myproblem/configs` directory as a json file named `pipe.json`. The file name can be specified using the `--name` (-n) option. 

By default, all six of the feature selection algorithms will be added, but this can be overidden using the `--featureselector` (-f) option to specify single algorithms. For example, if you just want to use SVM-RFE and iterative_MI, you would instead run:

```bash
$ chameleon-pipe features -d kfold_myproblem -f SVM-RFE -f iterative_MI
```

Currently, chameleon supports adding to the config file by running the above command multiple times when specifying an existing problem directory and config name. The only way to remove added parameters is to manually edit the config file.


### 4. Add/configure classifiers
Add classifiers to the pipeline in the config file using the following command:

```bash
$ chameleon-pipe classifiers -p kfold_myproblem/configs/pipe.json 
```

Where `-p` specifies the relative path to the config file. By default, every available classifier will be added to the pipeline, but this can be overriden by using the `--classifier` (-c) option to specify single classifiers. 


### 5. Run the test suite

Run the test suite with the config parameters with:

```bash
$ chameleon run -d kfold_myproblem -p kfold_myproblem/configs/pipe.json
```

*The folowing has not been implemented and may be subject to change*

By default, this will be inefficient and run each file + algorithm specified in the config one-by-one. This is usualy okay for most use cases, but could be result in extreme runtimes for some cases (e.g. SVM-RFE). The `--method` (-m) option  can improve on this, with `-m slurm` submitting each algorithm + classifier + file combination as a job through the Slurm batch system. 




## Chameleon Commands

This section describes all chameleon commands, sub-commands and
options.

Command | Description
| --- | --- |
`chameleon-data` | Import and prepare data for chameleon pipelines
`chameleon-pipe` | Configure pipeline 
`chameleon` | Run configured pipeline

### chamaleon-data

There are two subcommands: `chameleon-data format` and `chameleon-data kfold`.
#### format

Required:
Option | Argument | Description
| --- | --- | --- |
`--ftype` | `STRING` | Type of data file to format. Choices are `mat` and `arff`.
`--fpath` | `DIRECTORY` | Path to the raw data file.
`--name` | `STRING` | Name of output files e.g. '--name foo' will make files foo_X.pkl and foo_y.npy.

#### kfold
Required:
Option | Argument | Description
| --- | --- | --- | 
`--xfile` | `.PKL` | The name of the X data. e.g. '--Xfile proc_data/foo_X.pkl'
`--yfile` | `.NPY` | The name of the y data. e.g. '--yfile proc_data/foo_y.npy'
`--name` | `STRING` | Name of the output folder e.g. myproblem will create kfold_myproblem.

Optional:
Option | Argument | Default | Description
| --- | --- | --- | --- |
`--k` | `INT` | 5 |  The number of folds to partition the data".
`--random_seed` | `INT` | 666 |  Random state for assigning data to folds.
`--normalise/--no-normalise` |  | True |  Whether to normalise the data.
`--stratified/--not-stratified` |  | True |  Whether to apply stratified kfold.

### chameleon-pipe
There are three subcommands: `chameleon-pipe pipe`, `chameleon-pipe features`, and `chameleon-pipe classifiers`.

All available feature selection methods and classifiers are added to the pipeline by default. This can be overriden by specifying individual choices using the `--featureselector`/`-f` and `--classifier`/`-c` flags. Each flag can be used multiple times.

Arguments for `--featureselector`/`-f`:
Argument | Description
| --- | --- | 
`fischer` | Fischer score
`reliefF` | reliefF
`random-forest` | Random forest feature importance
`SVM-RFE` | SVM recursive feature elimination
`simple_MI` | Simple mutual information score
`iterative_MI` | Iterative mutual information selection

Arguments for `--classifier`/`-c`:
Argument | Description
| --- | --- | 
`naive-bayes` | Guassian Naive Bayes
`kNN` | k-nearest neighbours
`logistic-regression` | Logistic regession
`neural-net` | Neural network (multilayer perceptron)
`random-forest` | Random forest
`SVM` | Support vector machine (linear)

#### pipe
Required:
Option | Argument | Description
| --- | --- | --- |
`--data`/`-d` | `DIRECTORY` | The folder containing the prepared data e.g. kfold_myproblem.

Optional:
Option | Argument | Default | Description
| --- | --- | --- | --- |
`--featureselector`/`-f` | `STRING` |  |  Name of feature selection algorithms to add to the pipeline.
`--classifier`/`-c` | `STRING` |  |  Name of classifier algorithms to add to the pipeline.
`--name`/`-n` | `STRING` | pipe |  Name for the pipeline configuration file.

#### features
Required:
Option | Argument | Description
| --- | --- | --- |
`--data`/`-d` | `DIRECTORY` | The folder containing the prepared data e.g. kfold_myproblem.

Optional:
Option | Argument | Default | Description
| --- | --- | --- | --- |
`--featureselector`/`-f` | STRING |  |  Name of feature selection algorithms to add to the pipeline.
`--name`/`-n` | `STRING` | pipe |  Name for the pipeline configuration file.

#### classifiers
Required:
Option | Argument | Description
| --- | --- | --- |
`--configpath`/`-p` | `.JSON` | The path to the pipeline config file.

Optional:
Option | Argument | Description
| --- | --- | --- | 
`--classifier`/`-c` | `STRING` |  Name of classifier algorithms to add to the pipeline.

### chameleon
There is one subcommand: `chameleon run`.

Required:
Option | Argument | Description
| --- | --- | --- |
`--data`/`-d` | `DIRECTORY` | The folder containing the prepared data e.g. kfold_myproblem.
`--pipe`/`-p` | `.JSON` | The relative path to the pipe config file.

Optional:
Option | Argument | Default | Description
| --- | --- | --- | --- |
`--method`/`-m` | `STRING` | normal |  The method for running the program. Options are `normal` and `slurm`.
`--featureselection` | `BOOL` | True |  Whether to run feature selection.
`--predict` | BOOL | `True` |  Whether to run classification.
`--n_features`/`-n` | `INT` | 50 | Number of features to use in classifier predictions (the top 'n' features).
