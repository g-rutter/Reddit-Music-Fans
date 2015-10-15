# Reddit Music Fans

Predicting music tastes from unrelated interests. A small data science project.

## Introduction

Reddit user Stuck_In_the_Matrix [made available](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/) a database of every public Reddit comment. Kaggle picked up on this and are hosting the [comments from May 2015](https://www.kaggle.com/c/reddit-comments-may-2015), which I used here.

In this project, I compared the success of multiple machine learning algorithms under multiple conditions at classifying Reddit users as fans of one of **hiphop**, **rockmetal**, **electronic**, or **classical**, based on which subreddits _unrelated to music_ a user contributed to. A "fan" of hiphop is a user who posted in hiphop-related subreddits and not in subreddits of other genres.

I used Python 2 with the principal modules numpy, scipy, scikit-learn to learn from the data, sqlite3 to interact with the data, and matplotlib to produce visualisations.

#### The dataset

X has the shape (n_samples, n_predictors). Samples are music fans. Predictors are off-topic subreddits they do or do not post in.
Y has the shape (n_samples) and records the outcome, the fan's preferred genre.

The data is noisy, highly correlated and weakly predictive. There are approximately 65000 fans, and classical music has the smallest fanbase at ~1800 users. The top 2000 subreddits were tracked as predictors, and the average user posted in <1% of these, so the data is also sparse.

#### Files

* **main.py** - Driver. In its current state, will produce the graphical output in this README.
* **get_dataset_SQL.py** - Functions to create the X, Y database from SQL source.
* **manipulate_data.py** - Post-processing tools for the X, Y arrays.
* **subreddits.py** - Music subreddits by genre and exclusion list of all music subreddits.
* **music_2000offtopic.pickle** - Serialisation of the X, Y arrays and predictor labels (subreddit names) from SQL.

## Results
