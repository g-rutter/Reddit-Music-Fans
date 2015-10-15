# Reddit Music Fans

Predicting music tastes from unrelated interests.

## Introduction

This is a small data science project.

Reddit user Stuck_In_the_Matrix recently [made available](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/) a database of every public Reddit comment. Kaggle picked up on this and are hosting the [comments from May 2015](https://www.kaggle.com/c/reddit-comments-may-2015), which I used here.

In this project, I compared the success of multiple machine learning algorithms under multiple conditions at classifying Reddit users as fans of one of **hiphop**, **rockmetal**, **electronic**, or **classical**, based on which subreddits _unrelated to music_ a user contributed to. A fan of hiphop was defined as a user who posted in hiphop-related subreddits and in no subreddits corresponding to another genre.

I used Python 2 with the principal modules numpy, scipy, scikit-learn to learn from the data, sqlite3 to interact with the data, and matplotlib to produce visualisations.

#### The dataset

The data is noisy, highly correlated and weakly predictive. There are approximately 65000 fans, and classical music has the smallest fanbase at ~1800 users. The top 2000 subreddits were tracked as predictors, and the average user posted in <1% of these, so the data is also sparse.

#### Files

* **main.py** - Driver. In its current state, will produce the graphical output in this README.
* **get_dataset_SQL.py** - Functions to create the user-oriented database X, Y of music fan reddit users from the comment-oriented SQL source.
* **manipulate_data.py** - Post-processing tools for the X, Y arrays such as pruning, balancing and splitting the input into test and training data.
* **subreddits.py** - Subreddit data for this problem. Dict of subreddits corresponding to each musical genre, and list of all music-related subreddits to exclude from the predictors.

* **music_2000offtopic.pickle** - Serialisation of the X, Y arrays, along with labels for each predictor composing the columns of X. This is a checkpoint to skip the very slow complex queries to the large SQL database.

## Results
