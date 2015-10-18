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
from sklearn.lda import LDA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.grid_search import GridSearchCV
from sklearn.cluster import FeatureAgglomeration

from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import seaborn as sns

def temp_pickle(fn, topickle=None):

    if topickle != None:

        with open(fn, 'w') as data_f:
            pickle.dump(topickle, data_f)
            return topickle

    with open(fn, 'r') as data_f:
        return pickle.load(data_f)

def test_suite(algorithm, X, Y, train_thres=1, test_thresh=range(1,20)):
    (X_train, Y_train, X_test, Y_test) = \
            input_shuffle_split(X, Y, train=0.80, seed=0)

    print "Producing training data..."
    (X_train_3, Y_train_3) = prune_sparse_samples(X_train, Y_train,
                                                  threshold=train_thres,
                                                  silent=True)
    (X_train_bal_3, Y_train_bal_3) = balance_data(X_train_3, Y_train_3)

    print "Training algorithm..."
    fitted = algorithm.fit(X_train_bal_3.toarray(), Y_train_bal_3)

    print "Producing test data..."
    X_test_pruned = {}
    Y_test_pruned = {}
    for i in test_thresh:
        (X_test_pruned[i], Y_test_pruned[i]) =\
                prune_sparse_samples(X_test, Y_test, threshold=i, silent=True)
        (X_test_pruned[i], Y_test_pruned[i]) = balance_data(X_test_pruned[i], Y_test_pruned[i])
        print "Thresh", i, "score", fitted.A.score(X_test_pruned[i].toarray(), Y_test_pruned[i])


def plot_LDA_histogram(X_r3, X_r20, Yps3, Yps20):
    plt.figure(figsize=(8,4.5))
    ax = plt.subplot(1,1,1)
    plt.suptitle("Linear Discriminant Analysis: One-dimensional projection of data", size=14)
    plt.xlabel("Linear discriminant value", size=14)
    plt.ylabel("Probability density", size=14)

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

    plt.legend(fontsize=11)
    plt.savefig("README_figs/LDA_20vs3.svg")

if __name__ == "__main__":

    ##############
    #  Settings  #
    ##############

    pickle_filename = 'music_2000offtopic.pickle'

    fn = "LDA.pickle"

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

    # Delete those predictors I failed to exclude when I created the pickle.
    (X, nonmusic_subreddits) = remove_predictor(X, nonmusic_subreddits, music_subreddits)

    (X, Y, genres) = kill_outcome(X, Y, genres, 'classical')
    (X, Y, genres) = kill_outcome(X, Y, genres, 'electronic')

    ################################
    #  BRBM learning rate trainer  #
    ################################

    # Prune most irrelevant data just for slight speedup
    (X_5, Y_5) = prune_sparse_samples(X, Y, threshold=5)

    print "Training BRBM"

    rbm = BernoulliRBM(random_state=0, verbose=True, n_iter=20)
    logistic = LogisticRegression()

    rbm_logit = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    print GridSearchCV(rbm_logit,
                       param_grid={'rbm__learning_rate':(1,0.5,0.1,0.05,0.01,0.005,0.001)},
                       n_jobs=-1,
                       cv=4
                       ).fit(X_5, Y_5)\
                        .grid_scores_

    ###########################
    #  Feature agglomeration  #
    ###########################

    # agglo = FeatureAgglomeration(n_clusters=100,)
    # logistic = LogisticRegression()

    # agglo_logit = Pipeline(steps=[('agglo', agglo), ('logistic', logistic)])
    # test_suite(agglo_logit, X, Y, train_thres=1, test_thresh=[10])
