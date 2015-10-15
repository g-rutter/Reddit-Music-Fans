#!/usr/bin/env python
# -*- coding: utf-8 -*-

import bisect
import numpy as np

import sklearn as sk

def input_shuffle_split(X, Y, train=0.7):
    ''' Split input data into training and testing data
        WARNING: Once Y is shuffled, several preprocessing functions will no
                 longer work properly on it.

        Keyword argument:

        train -- fraction of data to assign to training.
    '''

    n_samples = len(Y)
    n_outcomes = Y[-1] - Y[0] + 1

    training_mask = np.zeros(n_samples, dtype=bool)

    # Iterate through each block of outcomes in the ordered outcome array Y
    # select samples randomly from within an outcome so that the final arrays
    # remain ordered.
    lower = 0
    for outcome_pop in count_contiguous_blocks(Y, n_outcomes):
        upper = outcome_pop + lower

        n_training_samples = int(outcome_pop*train)

        keep = sk.utils.shuffle(range(lower, upper))[:n_training_samples]
        training_mask[keep] = True

        lower = upper

    return (X[training_mask], Y[training_mask],
            X[training_mask == False], Y[training_mask == False])

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
    for outcome_pop in count_contiguous_blocks(Y, n_outcomes):
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

def summarise_dataset(X, Y, outcomes):
    print "--------------------------------------------"

    print "DATA SUMMARY\n"

    (n_samples, n_features) = X.get_shape()
    n_outcomes = Y[-1] - Y[0] + 1
    nnz = X.getnnz()

    print "{0:d} samples\n{1:d} features\n{2:.3f}% density\n"\
        .format(n_samples, n_features, 100.0*nnz/(n_samples*n_features))

    print "{0:d} outcomes:".format( n_outcomes )
    for i, block_length in enumerate(count_contiguous_blocks(Y, n_outcomes)):
        print " {0:7d} samples of \"{1:s}\"".format(block_length, outcomes[i])



    print "--------------------------------------------"

def kill_outcome(X, Y, outcomes, outcome):
    ''' Remove an outcome, provided as a string, from the dataset X, Y.
        Retains outcome-ordering of the dataset and removes gap left by killed
        outcome by decreasing index of all subsequent outcomes.
    '''

    try:
        i_outcome = outcomes.index(outcome)
    except IndexError:
        print "Outcome provided not found:", outcome
        print "Pass as string name of the outcome to exclude."
        exit ()

    n_outcomes = Y[-1] - Y[0] + 1
    mask = (Y != i_outcome)

    Y[Y > i_outcome] -= 1

    outcomes = list(outcomes)
    outcomes.remove(outcome)

    return X[mask], Y[mask], tuple(outcomes)

