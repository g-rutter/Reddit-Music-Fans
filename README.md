# Reddit Music Fans

Predicting music tastes from unrelated interests. A data science project.

## Introduction

**Aim:**

* Look at the success of **simple, interpretable models** to classify music fans on Reddit by their preferred music genre, based on which subreddits (Reddit subforums) _unrelated to music_ the user contributed to.
  * A _fan_ is defined as a user who posted in subreddits related to one genre and not in subreddits of other genres.
  * The problem is restricted to binary classification. Fans labelled **Hiphop** and **RockMetal** are classified, though **Classical** and **Electronic** were also tracked.
* Compare these simple approaches to one or two relatively complex or uninterpretable models, to see what gains in prediction accuracy can be made.

**Data source**: Reddit user Stuck_In_the_Matrix [made available](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/) a database of every public Reddit comment. Kaggle picked up on this and are hosting the [comments from May 2015](https://www.kaggle.com/c/reddit-comments-may-2015), which was used here.

**Coding tools:** Python 2 with modules numpy, scipy, scikit-learn, sqlite3 and matplotlib. Vim with tmux. IPython.

## Files

* **main.py** - Driver. In its current state, will produce the graphical output in this README.
* **get_dataset_SQL.py** - Functions to create the X, Y database from SQL source.
* **manipulate_data.py** - Post-processing tools for the X, Y arrays.
* **subreddits.py** - Music subreddits by genre and exclusion list of all music subreddits.
* **music_2000offtopic.pickle** - Prepared data from SQL. Takes hours to get this from the source.

## Data exploration

### Producing the input

The SQL source has 54,504,410 rows, and each is a unique comment. The columns are comment attributes, including user name and subreddit. After significant data processing, an array of features `X` was produced. Its rows are music fans, and its columns are subreddits unrelated to music. Each cell takes a boolean value; `True` if the fan posts in the subreddit. The top 2000 subreddits were tracked as features. An array `Y` records the outcomes; each fans preferred genre. The data was balanced on outcome in order to tackle the interesting case of maximum information entropy.

### Low-dimensional visualisation

Below, the data is shown projected onto a single linear component, the direction of maximum class separation, found by Linear Discriminant Analysis (LDA).

<p align="center"><img src ="https://cdn.rawgit.com/g-rutter/Reddit-Music-Fans/master/README_figs/LDA_20vs1.svg" /></p>

**Two datasets are shown, each with their own linear discriminant:** one is the full set, with all fans who posted in any of the top 2000 non-music subreddits, and the other only shows fans who posted in at least 20 non-music subreddits. The full set has 17,878 fans per class, but the ≥20 set has just 1,795.

**Key insights:**

* The class distributions have a single mode and are approximately Gaussian along this linear component, with differing means. This difference between classes highlights that there is information about music taste encoded in fans' unrelated interests.
* Classification in the ≥20 set is about 90% accurate by eye, which supports the applicability of linear modelling to this problem. This places an extremely optimistic upper limit of 90% on LDA classification accuracy for the ≥20 set, since the test data was also the training data.
* The lack of distinction between the classes on the full dataset hints that the best algorithm may only be a humble improvement from the 50% coin-flipping approach, for the less prolific Reddit posters.

### Features (predictors) of the dataset

The features are:

* **Highly correlated and anticorrelated** (because features are proxies for users' interests)

    This hurts model interpretability. Solutions will not be unique and interactions between predictors can produce unintuitive results.

* **Sparse**

    The most popular subreddit was posted in by 42.18% of fans, but the mean and median by just 0.42% and 0.14%, respectively. The lack of training examples may cause difficulty extracting useful information from these less popular subreddits.

* **Numerous** (high-dimensional problem)

    The configuration space of 2000 binary variables accepts <img src="http://www.sciweavers.org/tex2img.php?eq=2%5E%7B2000%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="2^{2000}" width="47" height="18" /> unique states. Depending on the relative weights of these states, we should expect each unseen sample to be novel. A non-parametric algorithm may struggle with this. Linear models may be able to fit an approximate solution, if linearity can approximate the target function.

The graph below shows how classification success with the standard logistic regression algorithm varies as increasingly sparse predictors are introduced. Note that the secondary Y-axis is truncated. The algorithm makes no accuracy gains past the first 40% most popular subreddits, despite the fact that about 5% of fans don't post in these. It is unlikely that the latter 60% of subreddits provide no information. Instead, model bias is likely being traded off for variance, as the less important features increase the model complexity.

<p align="center"><img src ="https://cdn.rawgit.com/g-rutter/Reddit-Music-Fans/master/README_figs/plot_sparsity.svg" /></p>

These observations on the features suggest the applicability of **feature agglomeration** for simplifying the linear model, with minimal loss of accuracy. Well-executed feature agglomeration limits the complexity that sparse predictors introduce to the model, without complete loss of their predictive power. Dimensionality reduction will enhance model interpretability by making solutions unique and stable and highlighting the important features.

### Model choices

**Agglomeration scheme**: A simple scheme was used. The list of features was sorted by [phi coefficient](https://en.wikipedia.org/wiki/Phi_coefficient) with the outcome. The list was sliced into N groups. Within groups, features were summed to produce a single integer value for each feature in each sample. Logistic regression was then used to fit this lower-dimensional dataset.

The scheme is equivalent to running logistic regression on the entire dataset while fixing the features within a group to have the same coefficient. Improvements to the scheme could include placing important or popular subreddits in smaller groups (or outside of groups) and grouping features which are correlated with each other _and_ the outcome, to mimimise information loss. However, the choice made is extremely straightforward, and may produce a simple, stable model with similar accuracy.

## Model performance

<p align="center"><img src ="https://cdn.rawgit.com/g-rutter/Reddit-Music-Fans/master/README_figs/agglo_logit.svg" /></p>
