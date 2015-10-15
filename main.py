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

import matplotlib.pyplot as plt

if __name__ == "__main__":

    ##############
    #  Settings  #
    ##############

    pickle_filename = 'music_old.pickle'

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

    (X, Y, genres) = kill_outcome(X, Y, genres, 'classical')

    (X, Y, nonmusic_subreddits) = prune_sparse_predictors(
                X, Y, nonmusic_subreddits, threshold=20)
    (X, Y) = prune_sparse_samples(X, Y, threshold=3)
    (X, Y) = balance_data(X, Y)

    # (X_train, Y_train, X_test, Y_test) = input_shuffle_split(X, Y, train=0.7)
    summarise_dataset(X, Y, genres)


    ######################
    #  Machine learning  #
    ######################


    classifiers = {
        # 'RForest'    : RandomForestClassifier(n_jobs=-1),
        # # 'ExtraTrees' : ExtraTreesClassifier(),
        # # 'AdaBoost'   : AdaBoostClassifier(),
        'LinearSV'   : LinearSVC(),
        # # 'GaussNB'    : GaussianNB(),  # Really bad. (BC data is correlated?)
        # # 'LinearDA'   : LDA(),
    }

    for name, classifier in classifiers.items():
        print "Training", name
        classifier.fit(X, Y)
        print "Test score = {0:.3f}".format( classifier.score(X.toarray(), Y) )

    print "Most relevant subreddits:"
    n_max = 60
    genretopsubs = {}
    genretopsubs[0] = []
    genretopsubs[1] = []
    genretopsubs[2] = []
    # genretopsubs[3] = []
    for name, classifier in classifiers.items():
        for i_max in range(n_max):
            for i_genre, j_subreddit in enumerate(classifier.coef_.argmax(axis=1)):
                classifier.coef_[i_genre][j_subreddit] = 0.0
                genretopsubs[i_genre].append( nonmusic_subreddits[j_subreddit] )

    for i_genre, listofsubs in genretopsubs.items():
        print genres[i_genre]
        print listofsubs

        # print classifier.get_params()

    ##############################
    #  Dimensionality reduction  #
    ##############################

    # print "Doing PCA"
    # pca = PCA(n_components=2)
    # X_r1 = pca.fit(X.toarray()).transform(X.toarray())

    # print "Doing LDA"
    # lda1 = LDA(n_components=1)
    # lda2 = LDA(n_components=2)
    # X_r1 = lda1.fit(X.toarray(), Y).transform(X.toarray())
    # X_r2 = lda2.fit(X.toarray(), Y).transform(X.toarray())

    # # PCA
    # # plt.figure()
    # # for i, c, genre in zip([0, 1, 2, 3], "rgby", genres):
        # # plt.scatter(X_r1[Y == i, 0], X_r1[Y == i, 1], c=c, label=genre)
        # # plt.legend()
        # # plt.title('PCA')

    # # LDA
    # plt.figure()
    # for i, c, genre in zip([0, 1, 2, 3], "rgby", genres):
        # plt.figure(0)
        # plt.scatter(X_r2[Y == i, 0], X_r2[Y == i, 1], c=c, label=genre)
        # plt.figure(1)
        # plt.hist(X_r1[Y == i], bins=25, histtype='step', color=c, label=genre)

    # plt.figure(0)
    # plt.legend()
    # plt.figure(1)
    # plt.legend()

    # plt.show()
