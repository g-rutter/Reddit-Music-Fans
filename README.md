# Reddit Music Fans

Predicting music tastes from unrelated interests. A data science project.

## Introduction

**Aim:**

* Classify music fans on Reddit by their preferred music genre, based on which subreddits (Reddit subforums) _unrelated to music_ the user contributed to.
  * A _fan_ is defined as a user who posted in subreddits related to one genre and not in subreddits of other genres.
  * The problem is restricted to binary classification. Fans labelled **Hiphop** and **RockMetal** are classified, though **Classical** and **Electronic** were also tracked.
* Use the model to learn about Reddit communities. Therefore, **interpretability takes precedence** over prediction accuracy.

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

**Two datasets are shown, each with their own linear discriminant:** one is the full set, with all fans who posted in any of the top 2000 non-music subreddits, and the other only shows fans who posted in at least 20 non-music subreddits. The full set has 12,829 fans per class, but the ≥20 set has just 1,795.

**Key insights:**

* The class distributions have a single mode and are approximately Gaussian along this linear component, with differing means. This difference between classes suggests that there is information about music taste encoded in fans' unrelated interests.
* Classification in the ≥20 set is about 90% accurate by eye, which supports the applicability of linear modelling to this problem. This places an optimistic upper limit of 90% on LDA classification accuracy for the ≥20 set, since the test data was also the training data.
* The lack of distinction between the classes on the full dataset hints that the best algorithm may only be a humble improvement from the 50% coin-flipping approach, for the less prolific Reddit posters.

### Correlation and sparsity of features

The features are:

* **Highly correlated and anticorrelated**

    The features are proxies for users' interests, so strong correlation is expected. This hurts model interpretability, since solutions will not be unique and interactions between predictors can produce unintuitive results.

* **Sparse and numerous**

    The most popular subreddit is posted in by 42.18% of fans, but the median by just 0.14%.

<p align="center"><img src ="https://cdn.rawgit.com/g-rutter/Reddit-Music-Fans/master/README_figs/plot_sparsity.svg" /></p>
