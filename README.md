# Reddit Music Fans

Predicting music tastes from unrelated interests. A small data science project.

## Introduction

**Aim:**

* Classify music fans on Reddit by their preferred music genre, based on which subreddits (Reddit subforums) _unrelated to music_ the user contributed to.
  * A _fan_ is defined as a user who posted in subreddits related to one genre and not in subreddits of other genres.
  * To keep to two outcomes, I classify fans labelled as **hiphop** and **rockmetal** here, though **classical** and **electronic** were also tracked.
* Use the model to learn about Reddit communities. Therefore, interpretability takes precedence over prediction accuracy.

**Data source**: Reddit user Stuck_In_the_Matrix [made available](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/) a database of every public Reddit comment. Kaggle picked up on this and are hosting the [comments from May 2015](https://www.kaggle.com/c/reddit-comments-may-2015), which I used here.

**Coding tools:** Python 2 with modules numpy, scipy, scikit-learn, sqlite3 and matplotlib. Vim with tmux. IPython.

### The dataset

The SQL source has 54,504,410 rows, and each is a unique comment. The columns are comment attributes, including user name and subreddit. After significant data processing, an array of predictors `X` was produced. Its rows are music fans, and its columns are subreddits unrelated to music. Each cell takes a boolean value; `True` if the user posts in the subreddit. `Y` records the outcomes; each fan's preferred genre.

The data is noisy, highly correlated and, we should assume, weakly predictive. There are approximately 65,000 fans, and classical music has the smallest fanbase at ~1,800 users. The top 2000 subreddits were tracked as predictors. The average user posted in <1% of these, so the data is also sparse. 

### Files

* **main.py** - Driver. In its current state, will produce the graphical output in this README.
* **get_dataset_SQL.py** - Functions to create the X, Y database from SQL source.
* **manipulate_data.py** - Post-processing tools for the X, Y arrays.
* **subreddits.py** - Music subreddits by genre and exclusion list of all music subreddits.
* **music_2000offtopic.pickle** - Serialisation of the X, Y arrays and predictor labels (subreddit names) from SQL.

## Results
