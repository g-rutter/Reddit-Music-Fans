# Reddit Music Fans

## What is this?

Reddit user [Stuck_In_the_Matrix](https://www.reddit.com/user/Stuck_In_the_Matrix) recently [made available](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/) a database of every public Reddit comment. Kaggle picked up on this and are hosting the [comments from May 2015](https://www.kaggle.com/c/reddit-comments-may-2015) for its community of data scientists to play with.

Here, I compare the success of multiple machine learning algorithms at classifying Reddit users as fans of one of **hiphop**, **rockmetal**, **electronic**, or **classical**, based on which subreddits _unrelated to music_ a users contributes to. A fan of hiphop is defined as a user who posted in hiphop-related subreddits and in no subreddits corresponding to another genre.

I use Python 2 with the principal modules numpy, scipy, scikit-learn and sqlite3 to interact with the data.
