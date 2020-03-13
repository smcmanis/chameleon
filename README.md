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


### 3. ...
Now we can run feature selection and classification with the `chameleon` command. The subcommands `feat` and `trainpredict` both start a command line interfaces (CLI) where we can set parameters, then perform feature selection and classification, respectively. The `pipe` subcommand will do this all in one. The parameters include things such as the datasets, feature selection algorithms, classifiers, etc.
Flag --normalise /--no-normalise


```bash
$ chameleon pipe make --data kfold_foo_newproblem --problem
Select options
= 

```


