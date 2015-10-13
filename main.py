#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Project to classify reddit user music fans based on which subreddits unrelated
to music they posted on. Uses the May2015 Reddit comments, from Kaggle.
'''

# Project-specific modules
from get_dataset_SQL import get_dataset_SQL
from subreddit_lists import subreddit_genres, genres, music_subreddits

import numpy as np
import pickle
import bisect

import sklearn as sk
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.qda import QDA
from sklearn.lda import LDA
from sklearn.pipeline import Pipeline

def input_shuffle_split(X, Y, train=0.7):
    ''' Split input data into training and testing data
        WARNING: Once Y is shuffled, several preprocessing functions will no
                 longer work properly on it.

        Keyword argument:

        train -- fraction of data to assign to training.
    '''

    n_samples = len(Y)
    n_training_samples = int(n_samples*train)

    # Shuffle both arrays identically through same seed.
    (X, Y) = sk.utils.shuffle(X, Y)

    return (X[:n_training_samples], Y[:n_training_samples],
            X[n_training_samples:], Y[n_training_samples:])

def prune_sparse_samples(X, Y, threshold=5):
    ''' Remove samples who posted on less than 'threshold' offtopic subreddits
    '''

    # Sorted array of row values with a nonzero entry
    # We will count up the duplicate entries and see if they beat the threshold
    occupied_rows = X.nonzero()[0]
    # Sort not needed here, but this is not guarenteed by the standard:
    occupied_rows.sort()
    n_samples = Y.shape[0]

    # Mark smallest segments for deletion
    block_start = 0
    delete = []
    block_lengths = count_contiguous_blocks(occupied_rows, n_samples)

    for i, block_length in enumerate(block_lengths):
        if block_length < threshold:
            delete.append(i)

    # Mask effectively removing those segments marked for deletion.
    mask = np.ones(n_samples, dtype=bool)
    mask[delete] = False

    print "{0:.2f}% of {1:d} samples pruned (threshold {2:d})"\
        .format((100.0*len(delete))/n_samples, n_samples, threshold)

    return X[mask], Y[mask]

def prune_sparse_predictors(X, Y, predictor_labels, threshold=10):
    ''' Remove predictors with fewer than 'threshold' nonzero samples. '''

    # Array of col values with a nonzero entry
    # We will count up the duplicate entries and see if they beat the threshold
    occupied_cols = X.nonzero()[1]
    occupied_cols.sort()
    n_predictors = X.get_shape()[1]

    # Mark smallest segments for deletion
    block_start = 0
    delete = []
    block_lengths = count_contiguous_blocks(occupied_cols, n_predictors)

    for i, block_length in enumerate(block_lengths):
        if block_length < threshold:
            delete.append(i)

    # Mask effectively removing those segments marked for deletion.
    mask = np.ones(n_predictors, dtype=bool)
    mask[delete] = False

    print "{0:.2f}% of {1:d} predictors pruned (threshold {2:d})"\
        .format((100.0*len(delete))/n_predictors, n_predictors, threshold)

    return X[:,mask], Y, predictor_labels[mask]

def balance_data(X, Y):
    ''' Finds least common outcome i and randomly removes other outcomes'
        samples so that all all outcome have the population of i.
    '''

    n_outcomes = Y[-1] - Y[0] + 1
    n_samples = Y.shape[0]

    # Get least common outcome and all outcomes' occurrences
    min_outcome_pop = len(Y)
    outcome_pops = []
    for i, outcome_pop in enumerate(count_contiguous_blocks(Y, n_outcomes)):
        outcome_pops.append(outcome_pop)
        if outcome_pop < min_outcome_pop:
            min_outcome_pop = outcome_pop

    # Prune other outcomes to match
    lower = 0
    mask = np.zeros(n_samples, dtype=bool)
    for i in range(n_outcomes):
        upper = outcome_pops[i] + lower

        # Select min_outcome_pop entries on [lower, upper) to keep.
        keep = sk.utils.shuffle(range(lower, upper))[:min_outcome_pop]
        #np.array of values from np.random
        mask[keep] = True

        print lower, upper
        lower = upper

    return X[mask], Y[mask]

def count_contiguous_blocks(A, n):
    ''' For a sorted array of blocks of ints from 0 to n-1,
        return generator of block length for each int.
    '''

    # Slide along sorted array once in O(n) time, yielding length of each number's segment
    block_start = 0
    for i in range(n):
        block_length = bisect.bisect(A[block_start:], i)
        block_start += block_length
        yield block_length

def summarise_dataset(X, Y, genres):
    print "--------------------------------------------"

    print "DATA SUMMARY\n"

    (n_samples, n_features) = X.get_shape()
    n_outcomes = Y[-1] - Y[0] + 1
    nnz = X.getnnz()

    print "{0:d} samples\n{1:d} features\n{2:.3f}% density\n"\
        .format(n_samples, n_features, 100.0*nnz/(n_samples*n_features))

    print "{0:d} outcomes:".format( n_outcomes )
    for i, block_length in enumerate(count_contiguous_blocks(Y, n_outcomes)):
        print " {0:7d} samples of \"{1:s}\"".format(block_length, genres[i])



    print "--------------------------------------------"

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
    (X, Y, nonmusic_subreddits) = prune_sparse_predictors(X, Y, nonmusic_subreddits, threshold=20)
    (X, Y) = prune_sparse_samples(X, Y, threshold=3)
    (X, Y) = balance_data(X, Y)
    summarise_dataset(X, Y, genres)

    print ("")

    (X_train, Y_train, X_test, Y_test) = input_shuffle_split(X, Y, train=0.7)

    ######################
    #  Machine learning  #
    ######################


    classifiers = {
        # 'RForest'    : RandomForestClassifier(n_jobs=-1),
        # 'ExtraTrees' : ExtraTreesClassifier(),
        # 'AdaBoost'   : AdaBoostClassifier(),
        'LinearSV'   : LinearSVC(),
        # 'GaussNB'    : GaussianNB(),  # Really bad. (BC data is correlated?)
        # 'LinearDA'   : LDA(),
        # 'pipe1'      : Pipeline([ ('feature_selection', LinearSVC()),
                                  # ('classification', LDA())
                               # ])
    }

    for name, classifier in classifiers.items():
        print "Training", name
        classifier.fit(X_train.toarray(), Y_train)
        print "Test score = {0:.3f}".format( classifier.score(X_test.toarray(), Y_test) )

    print "Most relevant subreddits:"
    n_max = 60
    genretopsubs = {}
    genretopsubs[0] = []
    genretopsubs[1] = []
    genretopsubs[2] = []
    genretopsubs[3] = []
    for name, classifier in classifiers.items():
        for i_max in range(n_max):
            for i_genre, j_subreddit in enumerate(classifier.coef_.argmax(axis=1)):
                classifier.coef_[i_genre][j_subreddit] = 0.0
                genretopsubs[i_genre].append( nonmusic_subreddits[j_subreddit] )

    for i_genre, listofsubs in genretopsubs.items():
        print genres[i_genre]
        print listofsubs

        # print classifier.get_params()
