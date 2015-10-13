# Reddit Music Fans

Predicting music tastes from unrelated interests.

## Introduction

This is a small data science project.

Reddit user Stuck_In_the_Matrix recently [made available](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/) a database of every public Reddit comment. Kaggle picked up on this and are hosting the [comments from May 2015](https://www.kaggle.com/c/reddit-comments-may-2015), which I use here.

In this project, I compare the success of multiple machine learning algorithms under multiple conditions at classifying Reddit users as fans of one of **hiphop**, **rockmetal**, **electronic**, or **classical**, based on which subreddits _unrelated to music_ a users contributes to. A fan of hiphop is defined as a user who posted in hiphop-related subreddits and in no subreddits corresponding to another genre.

I use Python 2 with the principal modules numpy, scipy, scikit-learn to learn from the data, sqlite3 to interact with the data, and matplotlib to produce visualisations.

### The dataset

The data is noisy, highly correlated and weakly predictive. There are approximately 65000 fans, and classical music has the smallest fanbase at ~1800 users. The top 2000 subreddits are tracked as predictors, and the average user posts in <1% of these, so the data is also sparse.
