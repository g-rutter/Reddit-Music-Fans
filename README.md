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

The SQL source has 54,504,410 rows, and each is a unique comment. The columns are comment attributes, including user name and subreddit. After significant data processing, an array of features `X` was produced. Its rows are music fans, and its columns are subreddits unrelated to music. Each cell takes a boolean value; `True` if the fan posts in the subreddit. `Y` records the outcomes; each fans preferred genre.

The data is noisy, highly correlated and sparse. There are approximately 47,000 fans of the two genres. To balance the two classes, the RockMetal class was slightly under-sampled. The top 2000 subreddits were tracked as features. The average user posted in <1% of these.

### Low-dimensional visualisation

Below, the data is shown projected onto a single linear component, the direction of maximum class separation, found by Linear Discriminant Analysis (LDA).

<p align="center"><img src ="https://cdn.rawgit.com/g-rutter/Reddit-Music-Fans/master/README_figs/LDA_20vs3.svg" /></p>

**Two datasets are shown, each with their own linear discriminant:** one of fans who posted in at least 3 off-topic subreddits, and another only of fans who posted in at least 20. The ≥3 set has 12,829 fans per class, but the ≥20 set has just 1,795.

**Key insights:**

* The class distributions have a single mode and are approximately Gaussian along this linear component.
* Classification in the ≥20 set is about 90% accurate by eye, which supports the applicability of linear modelling to this problem. This places an optimistic upper limit of 90% on LDA classification accuracy, since the test data was also the training data.
* The poor performance on the ≥3 dataset hints that the best algorithm will only be a humble improvement from the 50% coin-flipping approach, for the less prolific Reddit posters.

### Sample features

The features of the data are subreddits, and each cell takes a boolean value for whether the fan posts in the subreddit. The features are expected to be difficult for several reasons:

* **Highly correlated and/or anticorrelated** unstable for many algorithms, low interpretability because of multiple equivalent solutions
* **Very sparse** no small number of features will have broad predictive power as must users dont use most subreddits. need to group features or something, not SELECT some (and throw away others)
* **All features contain some information** very few feautes are totally non-predictive. Subreddits that are rarely posted in have low p-value, but taken together still contrinbute to model accuracy.

Could group similar features somehow to overcome correlation issues 
