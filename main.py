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
from make_graphs import *

import pickle

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.lda import LDA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.grid_search import GridSearchCV
from sklearn.cluster import FeatureAgglomeration
from sklearn.linear_model import RidgeClassifier

from joblib import Parallel, delayed

from scipy.stats import pearsonr

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



def two_stage_agglomerate(X, Y, predictor_labels, N=5, M=4):

    (n_samples, n_predictors) = X.shape
    Xar = X.toarray()
    corr = np.empty( shape=n_predictors, dtype=float )

    for i in range(n_predictors):
        corr[i] = pearsonr(Xar[:,i], Y)[0]

        # Turn each item in predictor labels into a list for grouping later.
        predictor_labels[i] = np.array([predictor_labels[i]])

    orderedindices = corr.argsort()
    corr_groups = np.array_split(orderedindices, N)

    X_new = np.empty(shape=(n_samples, N*M), dtype=int)
    predictor_labels_new = np.empty(shape=N*M, dtype=object)

    print X.shape, X_new.shape

    for i, group in enumerate(corr_groups):
        X_new[:,i*M:(i+1)*M], predictor_labels_new[i*M:(i+1)*M] =\
            group_by_correlation(Xar[:,group], predictor_labels[group], M)

    return X_new, predictor_labels_new

def group_by_correlation(X, predictor_labels, M):
    ''' Get correlation between predictors (columns) of array X
        ONLY SETS i<j TERMS IN CORR[i][j]

        Time complexity O(n^2)

        M is number of groups to group columns into.
    '''

    try:
        X = X.toarray(dtype=int)
    except AttributeError:
        pass

    n_predictors = X.shape[1]

    if M == 1:
        #Merge all columns, no need to track correlation
        X = X.astype(int, copy=False).sum(axis=1)
        predictor_labels[0] = np.array([label for label in predictor_labels])
        predictor_labels = predictor_labels[0:1]
        return np.array([X]).transpose(), predictor_labels

    print ("NEEDS FIXING HOW X COLS ARE MERGED.")

    # Compute covariance matrix
    corr = np.empty(shape=(n_predictors,n_predictors))
    corr.fill(-1.0)
    for j in range(n_predictors):
        for i in range(j):
            corr[i][j] = pearsonr(X[:,j], X[:,i])[0]

    while n_predictors > M:

        # Group most collinear pair
        index = np.nonzero(corr == np.max(corr))
        a, b = index[0][0], index[1][0]

        X[:,a] += X[:,b]

        # Calculate new correlation terms for a
        for i in range(n_predictors):
            if i < a:
                corr[i][a] = pearsonr(X[:,a], X[:,i])[0]
            if i > a:
                corr[a][i] = pearsonr(X[:,a], X[:,i])[0]

        #group predictor label
        predictor_labels[a] = np.append(predictor_labels[a], predictor_labels[b])

        # Mask b in X, corr, and predictor label
        mask = np.ones([n_predictors], dtype=bool)
        mask[b] = False
        corr = corr[mask]
        corr = corr[:,mask]
        X = X[:,mask]
        predictor_labels = predictor_labels[mask]

        n_predictors = n_predictors - 1

    return X, predictor_labels

if __name__ == "__main__":

    ##############
    #  Settings  #
    ##############

    pickle_filename = 'pickles/music_2000offtopic.pickle'

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

    (X, Y, genres) = kill_outcome(X, Y, genres, 'classical')
    (X, Y, genres) = kill_outcome(X, Y, genres, 'electronic')

    # Delete those predictors I failed to exclude when I created the pickle.
    # Delete any predictors which are empty after killing outcomes
    (X, nonmusic_subreddits) = sanitise_predictors(X, nonmusic_subreddits,
                                                   music_subreddits)

    (Xps1, Yps1) = prune_sparse_samples(X, Y, threshold=1)
    # (Xps20, Yps20) = prune_sparse_samples(X, Y, threshold=20)

    (Xps1, Yps1) = balance_data(Xps1, Yps1)
    # (Xps20, Yps20) = balance_data(Xps20, Yps20)

    ################################
    #  BRBM learning rate trainer  #
    ################################

    # Prune most irrelevant data just for slight speedup
    # (X_5, Y_5) = prune_sparse_samples(X, Y, threshold=5)

    # print "Training BRBM"

    # fn = "pickles/trained_BernoulliRBM.pickle"

    # rbm = []
    # n_folds = 5

    # for i in range(5):
        # rbm.append( BernoulliRBM(random_state=i, verbose=True, n_iter=100, learning_rate=0.05) )

    # (n_samples_1, n_features) = Xps1.shape

    # kf = KFold(n_samples_1, n_folds=n_folds, shuffle=True, random_state=0)

    # for i, (train, test) in enumerate(kf):
        # rbm[i].fit(Xps1[train], Yps1[train])

    # with open(fn, 'w') as data_f:
        # pickle.dump(rbm, data_f)

    # logistic = LogisticRegression()

    # rbm_logit = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

    # print GridSearchCV(rbm_logit,
                       # param_grid={'rbm__learning_rate':(0.1,0.05)},
                       # n_jobs=-1,
                       # cv=4
                       # ).fit(X_5, Y_5)\
                        # .grid_scores_

    ###########################
    #  Feature agglomeration  #
    ###########################

    # #train
    # (X_train, Y_train, X_test, Y_test) = input_shuffle_split(X, Y)

    # print X_train.get_shape()

    # agglo = phi_agglomerate(N=15).fit(X_train, Y_train, nonmusic_subreddits)

    # print agglo

    # X_train_agglo, labels_agglo = agglo.transform(X_train)

    # print X_train_agglo.shape

    # (X_test, Y_test) = prune_sparse_samples(X_test, Y_test, threshold=10)
    # X_test_agglo, labels_aggo = agglo.transform(X_test)

    # print LogisticRegression().fit(X_train_agglo, Y_train).score(X_test_agglo, Y_test)

    ###########
    #  Plots  #
    ###########

    # plot_LDA_histogram(Xps1, Xps20, Yps1, Yps20)

    # plot_sparsity(Xps1,Yps1)

    plot_agglo_logit(Xps1, Yps1, nonmusic_subreddits)

    # newly_misclassified(Xps1, Yps1, nonmusic_subreddits)

    ################################
    #  Testing agglo reduce funcs  #
    ################################

    # Xps1 = Xps1.toarray()

    # logit = LogisticRegression()
    # (n_samples_1, n_features) = Xps1.shape

    # n_folds = 4
    # kf = KFold(n_samples_1, n_folds=n_folds, shuffle=True)

    # n_lo = 8
    # n_hi = 20
    # step = 2
    # n_groups_gen = range(n_lo, n_hi+1, step)

    # agglo_as = [0.0 for i in n_groups_gen]
    # agglo_bs = [0.0 for i in n_groups_gen]

    # for i_fold, (train, test) in enumerate(kf):
        # print i_fold

        # logit.fit(Xps1[train], Yps1[train])

        # for i, n_groups in enumerate( n_groups_gen ):

            # # agglo = phi_agglomerate(N=n_groups).fit(Xps1[train], Yps1[train])

            # # Xagglo_train_a, __ = agglo.transform(Xps1[train], nonmusic_subreddits)
            # # Xagglo_test_a, __ = agglo.transform(Xps1[test], nonmusic_subreddits)
            # # logit.fit(Xagglo_train_a, Yps1[train])
            # # agglo_as[i] += (100.0*logit.score(Xagglo_test_a, Yps1[test])/n_folds)

            # agglo = phi_agglomerate(N=n_groups).fit(Xps1[train], Yps1[train])

            # Xagglo_train_b, __ = agglo.transform(Xps1[train], nonmusic_subreddits)
            # Xagglo_test_b, __ = agglo.transform(Xps1[test], nonmusic_subreddits)
            # logit.fit(Xagglo_train_b, Yps1[train])
            # agglo_bs[i] += (100.0*logit.score(Xagglo_test_b, Yps1[test])/n_folds)

    # # plt.plot(n_groups_gen, agglo_as, label="no norm")
    # plt.plot(n_groups_gen, agglo_bs, label="norm")
    # plt.legend(fontsize=11)
    # plt.show()
