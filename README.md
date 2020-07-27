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
3. Configure the full pipeline with `chameleon pipe`. Alternatively, individually configure the pipeline using `chameleon features` to add feature selection algorithms and `chameleon classifiers` to add classifiers.
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
$ chameleon features -d kfold_myproblem 
```
Where `-d` specifies the directory containing your problem. This will create a pipeline configuration file that stores the data file names, feature selection algorithms, and classifiers. The config file is saved in the `kfold_myproblem/configs` directory as a json file named `pipe.json`. The file name can be specified using the `--name` (-n) option. 

By default, all six of the feature selection algorithms will be added, but this can be overidden using the `--featureselector` (-f) option to specify single algorithms. For example, if you just want to use SVM-RFE and iterative_MI, you would instead run:

```bash
$ chameleon features -d kfold_myproblem -f SVM-RFE -f iterative_MI
```

Currently, chameleon supports adding to the config file by running the above command multiple times when specifying an existing problem directory and config name. The only way to remove added parameters is to manually edit the config file.


### 4. Add/configure classifiers
Add classifiers to the pipeline in the config file using the following command:

```bash
$ chameleon classifiers -p kfold_myproblem/configs/pipe.json 
```

Where `-p` specifies the relative path to the config file. By default, every available classifier will be added to the pipeline, but this can be overriden by using the `--classifier` (-c) option to specify single classifiers. 


### 5. Run the test suite

Run the test suite with the config parameters with:

```bash
$ chameleon run -d kfold_myproblem -p kfold_myproblem/configs/pipe.json
```

*The folowing has not been implemented and may be subject to change*

By default, this will be inefficient and run each file + algorithm specified in the config one-by-one. This is usualy okay for most use cases, but could be result in extreme runtimes for some cases (e.g. SVM-RFE). The `--method` (-m) option  can improve on this, with `-m slurm` submitting each algorithm + classifier + file combination as a job through the Slurm batch system. 




