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
$ pip install /path/to/chameleon/.
```

or, if you plan on doing development

```bash
$ pip install -e /path/to/chameleon/.
```

## Outline

The basic steps in using Chameleon are:

1. Import and format a dataset with `chameleon-data format`,
2. Create *k*-fold cross-validation partitions with `chameleon-data kfold`,

*The folowing steps have not been implemented and may be subject to change*

3. Select and run feature selection models followed by prediction models with `chameleon-pipe`. Alternatively, run feature selection with `chameleon feat` and/or prediction with `chameleon trainpredict`.
5. Run performance evaluation and visualisation tools using `chameleon-results`

<!-- ![Pipeline workflow diagram](pipelinediagram.svg) -->

## Data Prerequisites

Since Chameleon is currently in the prototype stage, there are very strict requirements for data input:

1. Covariates and targets must be in one of the following formats:
    - `.mat` - A single file in the same format as the biological datasets seen [here](http://featureselection.asu.edu/datasets.php).
    - `.arff` - A single file with the targets as the last attribute.
2. All covariates must be continuous.
3. Targets must be binary.
4. No missing values.

## Usage Example

<!-- Before starting, we must make sure that the data files we want to import are located in the `/path/to/chameleon/data/external` directory.  -->

### 1. Import the data

Say we have a `.mat` file located at `/path/to/data.mat`. First, we need to extract the data from this file so it is in the correct format for chameleon. 

```bash
$ chameleon-data format --ftype mat --fpath /path/to/data.mat --name foo
```

This command will produce two files. One for the covariate data `X_foo.pkl` and another for the targets `y_foo.npy`. They will be located in the `proc_data` folder of the current working directory. The folder will be created if it doesn't already exist.


### 2. Create cross-validation partitions

Chameleon currently uses *k*-fold cross-validation, so we are now going to randomly partition the data into *k* folds to make *k* train/test sets.

```bash
$ chameleon-data kfold --xfile proc_data/X_foo.pkl --yfile proc_data/y_foo.npy --name myproblem
```
The default *k* is 5, but can be set using `--k`. The *k*-fold method is stratified by default but can be suppressed using the `--not-stratified` flag. Similarly, the data will be normalised by default unless the `--no-normalise` flag is included.
This command will create a new folder in the current directory called `kfold_myproblem` that will contain *k* files in the format `fold1of*k*.pkl`. These files contain the unique sets of training and test data for the fold.


### 3. Configure feature selection choices
The chameleon utility provides a command line option for quickly setting up a config file for your new problem that specifies the feature selection algorithms and the classifiers that you want to run for each data file. 
The config file can be manually edited without issue.
To start, create the config file and specify the feature selection algorithms with the following command: 
 
```bash
$ chameleon-feature add -d kfold_myproblem 
```
Where `-d` specifies the directory containing your problem. This will create a dictionary with keys for each data .pkl file in the kfold_myproblem directory. Each key will have a list of associated feature selection algorithms and an empty list for optionally adding classifiers later. The config dictionary is saved in the `kfold_myproblem/configs` directory as a json file named `pipe.json`, but this file name can be specified using the `--name` option. 

By default, all six of the feature selection algorithms will be added, but this can be overidden using the `--algorithm` (-a) option to specify single algorithms. For example, if you just want to use SVM-RFE and iterative_MI, you would instead run:

```bash
$ chameleon-feature add -d kfold_myproblem -a SVM-RFE -a iterative_MI
```

Currently, chameleon supports adding to the config file by running the `c`hameleon-feature add` command multiple times, but the only way to remove added parameters is to manually edit the config file.


### 4. Optionally add/configure classifiers

*coming soon*

### 5. Run the test suite



*The folowing steps have not been implemented and may be subject to change*
