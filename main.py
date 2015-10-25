#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Project to classify reddit user music fans based on which subreddits unrelated
to music they posted on. Uses the May2015 Reddit comments, from Kaggle.
'''

# Project-specific modules
from get_dataset_SQL import get_dataset_SQL
from subreddits import subreddit_genres, genres, music_subreddits
from manipulate_data import sanitise_predictors, balance_data, kill_outcome,\
                            prune_sparse_samples
from make_graphs import plot_LDA_histogram, plot_sparsity, plot_agglo_logit,\
                        plot_RBM, graph_music_taste

import pickle
from numpy import array

# sklearn
from sklearn.neural_network import BernoulliRBM
from sklearn.cross_validation import KFold

def train_BRBMs(Xps1, Yps1, Xps20, Yps20):
    ''' Trains a set of BRBMs on the dataset and saves as pickles.
        Checks if pickles already exist to not repeat work on restart.
    '''

    ##############
    #  Settings  #
    ##############

    N_range = range(10, 141, 20)
    learning_rate = 0.05
    n_iter = 800
    n_folds = 4
    rand = 0

    ###########
    #  Train  #
    ###########

    #Unpruned samples first (Xps1, Yps1)
    (n_samples_1, n_features) = Xps1.shape
    kf = KFold(n_samples_1, n_folds=n_folds, shuffle=True, random_state=rand)

    for N in N_range:

        print "Now training for N =", N

        for i, (train, test) in enumerate(kf):
            filename = "pickles/BRBMs/N"+str(N)+"_f"+str(i)+".pickle"
            try:
                with open(filename, 'r') as data_f:
                    pass
            except IOError:
                rbm = BernoulliRBM(n_components=N, random_state=rand,
                                   verbose=True, n_iter=n_iter,
                                   learning_rate=learning_rate)
                rbm.fit(Xps1[train], Yps1[train])
                with open(filename, 'w') as data_f:
                    pickle.dump(rbm, data_f)

    #Pruned samples (Xps20, Yps20)
    (n_samples_20, n_features) = Xps20.shape
    kf = KFold(n_samples_20, n_folds=n_folds, shuffle=True, random_state=rand)

    for N in N_range:

        print "Now training for N =", N

        for i, (train, test) in enumerate(kf):
            filename = "pickles/BRBMs/N"+str(N)+"_f"+str(i)+"_ps20.pickle"
            try:
                with open(filename, 'r') as data_f:
                    pass
            except IOError:
                rbm = BernoulliRBM(n_components=N, random_state=rand,
                                   verbose=True, n_iter=n_iter,
                                   learning_rate=learning_rate)
                rbm.fit(Xps20[train], Yps20[train])
                with open(filename, 'w') as data_f:
                    pickle.dump(rbm, data_f)

if __name__ == "__main__":

    ##############
    #  Settings  #
    ##############

    XY_pickle = 'pickles/music_2000offtopic.pickle'

    #################################
    #  Data preparation or read-in  #
    #################################

    try:
        with open(XY_pickle, 'r') as dataset_file:
            (X, Y, nonmusic_subreddits) = pickle.load(dataset_file)
    except IOError:
        print "No pickle. Creating dataset from SQL source."
        db_file = "../database.sqlite"
        table_name = "May2015"
        (X, Y, nonmusic_subreddits) = get_dataset_SQL(db_file, table_name)

        print "Checkpointing: Saving pickle of output as "+XY_pickle
        with open(XY_pickle, 'w') as dataset_file:
            pickle.dump((X, Y, nonmusic_subreddits), dataset_file)

    ###################
    #  Preprocessing  #
    ###################

    X = X.tocsr()
    nonmusic_subreddits = array(nonmusic_subreddits, dtype=object)

    (X, Y, genres) = kill_outcome(X, Y, genres, 'classical')
    (X, Y, genres) = kill_outcome(X, Y, genres, 'electronic')

    # Delete those predictors I failed to exclude when I created the pickle.
    # Delete any predictors which are empty after killing outcomes
    (X, nonmusic_subreddits) = sanitise_predictors(X, nonmusic_subreddits,
                                                   music_subreddits)

    (Xps1, Yps1) = prune_sparse_samples(X, Y, threshold=1)
    (Xps20, Yps20) = prune_sparse_samples(X, Y, threshold=20)

    (Xps1, Yps1) = balance_data(Xps1, Yps1)
    (Xps20, Yps20) = balance_data(Xps20, Yps20)

    ################################
    #  BRBM learning rate trainer  #
    ################################

    # print "Training BRBM"

    # train_BRBMs(Xps1, Yps1, Xps20, Yps20)

    ###########
    #  Plots  #
    ###########

    plot_LDA_histogram(Xps1, Xps20, Yps1, Yps20)

    plot_sparsity(Xps1, Yps1)

    plot_agglo_logit(Xps1, Yps1, nonmusic_subreddits)

    plot_RBM(Xps1, Yps1)

    graph_music_taste(Xps1, Yps1, nonmusic_subreddits)
