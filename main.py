#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Project to classify reddit user music fans based on which subreddits unrelated
to music they posted on. Uses the May2015 Reddit comments, from Kaggle.
'''

# Project-specific modules
from get_dataset_SQL import get_dataset_SQL
from subreddits import subreddit_genres, genres, music_subreddits
from manipulate_data import *

import pickle

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.qda import QDA
from sklearn.lda import LDA
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns

def temp_pickle(topickle=None):

    fn = "temp.pickle"

    if topickle != None:

        with open(fn, 'w') as data_f:
            pickle.dump(topickle, data_f)
            return topickle

    with open(fn, 'r') as data_f:
        return pickle.load(data_f)


def plot_LDA_histogram(X_r3, X_r20, Yps3, Yps20):
    plt.figure(figsize=(10,5.5))
    ax = plt.subplot(1,1,1)
    plt.suptitle(u"Linear Discriminant Analysis: Fans who posted in ≥20 subreddits are more distinct by class than fans who posted in ≥3.")
    plt.xlabel("Linear discriminant value")
    plt.ylabel("Probability density")

    snscol = sns.color_palette("Set1", n_colors=8, desat=.5)

    # settings
    bins = 20
    linewidth=2
    labels=('RockMetal', 'Hiphop')

    i=0
    plt.hist(X_r3[Yps3 == i], normed=True, bins=bins, histtype='step', color=snscol[i], label=labels[i]+u' ( ≥3 subreddits)', linewidth=linewidth)
    i=1
    plt.hist(X_r3[Yps3 == i], normed=True, bins=bins, histtype='step', color=snscol[i], label=labels[i]+u' ( ≥3 subreddits)', linewidth=linewidth)
    i=0
    plt.hist(X_r20[Yps20 == i], normed=True, bins=bins, histtype='step', color=snscol[i], label=labels[i]+u' ( ≥20 subreddits)', linestyle=('dashed'), linewidth=linewidth)
    i=1
    plt.hist(X_r20[Yps20 == i], normed=True, bins=bins, histtype='step', color=snscol[i], label=labels[i]+u' ( ≥20 subreddits)', linestyle=('dashed'), linewidth=linewidth)

    plt.legend()
    plt.show()

if __name__ == "__main__":

    ##############
    #  Settings  #
    ##############

    pickle_filename = 'music_2000offtopic.pickle'

    #################################
    #  Data preparation or read-in  #
    #################################

    try:
        with open(pickle_filename, 'r') as dataset_file:
            (X, Y, nonmusic_subreddits) = pickle.load(dataset_file)
    except IOError:
        print ("No pickle. Creating dataset from SQL source.")
        db_file = "../database.sqlite"
        table_name = "May2015"
        (X, Y, nonmusic_subreddits) = get_dataset_SQL(db_file, table_name)

        print ("Checkpointing: Saving pickle of output as "+pickle_filename)
        with open(pickle_filename, 'w') as dataset_file:
            pickle.dump((X, Y, nonmusic_subreddits), dataset_file)

    ###################
    #  Preprocessing  #
    ###################

    X = X.tocsr()
    nonmusic_subreddits = np.array(nonmusic_subreddits, dtype=object)

    summarise_dataset(X, Y, genres)


    # Delete those predictors I failed to exclude when I created the pickle.
    (X, nonmusic_subreddits) = remove_predictor(X, nonmusic_subreddits, music_subreddits)

    (X, Y, genres) = kill_outcome(X, Y, genres, 'classical')
    (X, Y, genres) = kill_outcome(X, Y, genres, 'electronic')

    # (X, nonmusic_subreddits) = prune_sparse_predictors(
                # X, nonmusic_subreddits, threshold=20)

    (Xps20, Yps20) = prune_sparse_samples(X, Y, threshold=20)
    (Xps3, Yps3) = prune_sparse_samples(X, Y, threshold=3)

    (Xps20, Yps20) = balance_data(Xps20, Yps20)
    (Xps3, Yps3)   = balance_data(Xps3, Yps3)
    summarise_dataset(Xps20, Yps20, genres)

    try:
        (X_r3, X_r20, Yps3, Yps20) = temp_pickle()
    except (IOError, ValueError):

        # (X_train, Y_train, X_test, Y_test) = input_shuffle_split(X, Y, train=0.8)
        summarise_dataset(Xps3, Yps3, genres)
        summarise_dataset(Xps20, Yps20, genres)

        # (Xps20_train, Yps20_train, Xps20_test, Yps20_test) = input_shuffle_split(Xps20, Yps20, train=0.8)

        ##############################
        #  Dimensionality reduction  #
        ##############################

        print "Doing LDA"
        lda3 = LDA(n_components=1)
        lda20 = LDA(n_components=1)
        X_r3 = lda3.fit(Xps3.toarray(), Yps3).transform(Xps3.toarray())
        X_r20 = lda20.fit(Xps20.toarray(), Yps20).transform(Xps20.toarray())

        temp_pickle( topickle=(X_r3, X_r20, Yps3, Yps20) )

    plot_LDA_histogram(X_r3, X_r20, Yps3, Yps20)

    ##################################
    #  Logit on LDA-reduced 1D data  #
    ##################################

    # logit = LogisticRegression()

    # (X_r20_train, Y_r20_train, X_r20_test, Y_r20_test) = \
                # input_shuffle_split(X_r20, Yps20, train=0.8)
    # logit.fit(X_r20_train, Y_r20_train)
    # print "logit with LDA preprocessing score", logit.score(X_r20_test, Y_r20_test)

    # logit = LogisticRegression()

    # (Xps20_train, Yps20_train, Xps20_test, Yps20_test) = \
                # input_shuffle_split(Xps20, Yps20, train=0.8)
    # logit.fit(Xps20_train, Yps20_train)
    # print "logit score" , logit.score(Xps20_test, Yps20_test)

    #########################
    #  LDA on its own data  #
    #########################

    lda = LDA()
    # print "logit with LDA preprocessing score", logit.score(X_r20_test, Y_r20_test)
