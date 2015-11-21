#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Collection of functions to produce the figures in README_figs/, which
    appear in the top-level readme, intended to be read on github.
'''

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from matplotlib import cm

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.lda import LDA

import pickle
from itertools import combinations
# from graph_tool.all import Graph, graph_draw, radial_tree_layout

from manipulate_data import balance_data, prune_sparse_samples, phi_agglomerate

def plot_LDA_histogram(Xps1, Xps20, Yps1, Yps20):
    ''' Produce README_figs/LDA_20vs1.svg

        psN = pruned samples with threshold N
        rN = transformed coordinates from LDA using
             pruned samples with threshold N
    '''

    ##################
    #  Make dataset  #
    ##################

    pickle_fn = "pickles/plot_LDA_histogram.pickle"

    try:
        with open(pickle_fn, 'r') as data_f:
            (X_r1, X_r20) = pickle.load(data_f)

    except IOError:
        print "No pickle. Making dataset."

        X_r20 = LDA().fit_transform(Xps20.toarray(), Yps20)
        X_r1 = LDA().fit_transform(Xps1.toarray(), Yps1)

        topickle = (X_r1, X_r20)

        with open(pickle_fn, 'w') as data_f:
            pickle.dump(topickle, data_f)

    ##########
    #  Plot  #
    ##########

    # settings
    bins = 30
    linewidth = 2
    labels = ('RockMetal', 'Hiphop')
    labelfontsize = 14

    plt.figure(figsize=(8, 4.5))
    plt.subplot(1, 1, 1)
    plt.suptitle("Projection of samples along the linear discriminant", size=16)
    plt.xlabel("Linear discriminant value", size=labelfontsize)
    plt.ylabel("Probability density", size=labelfontsize)

    snscol = sns.color_palette("Set1", n_colors=8, desat=.5)

    i = 0
    plt.hist(X_r1[Yps1 == i], normed=True, bins=bins, histtype='step',
            color=snscol[i], label=labels[i], linewidth=linewidth)
    i = 1
    plt.hist(X_r1[Yps1 == i], normed=True, bins=bins, histtype='step',
             color=snscol[i], label=labels[i], linewidth=linewidth)
    i = 0
    plt.hist(X_r20[Yps20 == i], normed=True, bins=bins, histtype='step',
             color=snscol[i], label=labels[i]+u' (≥20 subreddits)',
             linestyle=('dashed'), linewidth=linewidth)
    i = 1
    plt.hist(X_r20[Yps20 == i], normed=True, bins=bins, histtype='step',
             color=snscol[i], label=labels[i]+u' (≥20 subreddits)',
             linestyle=('dashed'), linewidth=linewidth)

    plt.legend(fontsize=11)
    plt.savefig("README_figs/LDA_20vs1.svg")

def plot_sparsity(Xps1, Yps1):
    ''' Produce README_figs/plot_sparsity.xvg

        Takes the X and Yps1 input arrays with the empty samples pruned
        i.e. Xps1, Yps1 = prune_sparse_samples(X, Yps1, threshold=1)
    '''

    pickle_fn = "pickles/plot_sparsity.pickle"

    try:
        with open(pickle_fn, 'r') as data_f:
            (x1, cum_posters, x2, logit_scores) = pickle.load(data_f)

    except IOError:
        print "No pickle. Making dataset."

        Xps1 = Xps1.toarray()

        (n_samples, n_features) = Xps1.shape

        N_posters = Xps1.sum(axis=0)
        N_posters = np.asarray(N_posters)

        argsort_N_posters = N_posters.argsort()[::-1]
        ordered_Xps1 = Xps1[:, argsort_N_posters]

        seen_posters = np.zeros(n_samples, dtype=int)

        kf = KFold(n_samples, n_folds=4, shuffle=True)

        logit_scores = []
        x1 = []
        x2 = []
        cum_posters = []

        for i in range(1, n_features):

            progress = float(100*i)/n_features
            x1.append(progress)

            # Cumulative users
            feature_posters = ordered_Xps1[:, i]
            seen_posters = np.logical_or(seen_posters, feature_posters)
            cum_posters.append(100.0*seen_posters.sum()/n_samples)

            # Classification accuracy
            if i % 4 == 0:

                x2.append(progress)
                logit_score = 0.0

                for train, test in kf:
                    logit_score += LogisticRegression()\
                                    .fit(ordered_Xps1[train, :i], Yps1[train])\
                                    .score(ordered_Xps1[test, :i], Yps1[test])

                logit_scores.append(100.0*logit_score/4)

        topickle = (x1, cum_posters, x2, logit_scores)

        with open(pickle_fn, 'w') as data_f:
            pickle.dump(topickle, data_f)

    linewidth = 2
    fontsize = 14

    plt.figure(figsize=(8, 4.5))
    plt.suptitle("Diminishing returns with inclusion of sparse features",
                 size=16)

    snscol = sns.color_palette("Set1", n_colors=8, desat=.5)
    host = host_subplot(111, axes_class=AA.Axes)

    par1 = host.twinx()

    host.set_xlim(0, 100)
    host.set_ylim(0, 100)

    host.set_xlabel("Included subreddits (%; descending popularity)")
    host.set_ylabel("Fans with at least one post (%)")
    par1.set_ylabel("Logistic regression accuracy (%)")

    host.plot(x1, cum_posters, label="Cumulative", color=snscol[0],
              linewidth=linewidth)
    par1.plot(x2, logit_scores, label="Logit", color=snscol[1],
              linewidth=linewidth)

    par1.set_ylim(50, 75)

    # host.legend()

    host.axis["bottom"].label.set_fontsize(fontsize)
    host.axis["left"].label.set_fontsize(fontsize)
    par1.axis["right"].label.set_fontsize(fontsize)

    host.axis["left"].label.set_color(snscol[0])
    par1.axis["right"].label.set_color(snscol[1])

    plt.draw()
    plt.savefig("README_figs/plot_sparsity.svg")

def plot_agglo_logit(Xps1, Yps1, nonmusic_subreddits):
    ''' Creates file README_figs/agglo_logit.svg by training logit regression
        to work with various sizes of predictor groups and measuring model
        accuracy and parameter fluctuation.

        Uses helper functions agglo_logit_calc() to fit and run the models,
        and get_mrmsd() to produce fluctuation dataset.
    '''

    ##############
    #  Get data  #
    ##############

    pickle_fn = "pickles/agglo_logit.pickle"

    try:
        with open(pickle_fn, 'r') as data_f:
            (n_lo, n_hi, logit_1, logit_20, n_groups_gen,
             agglo_1s, agglo_20s, params, logit_params) = pickle.load(data_f)

    except IOError:
        print "No pickle. Making dataset."

        topickle = agglo_logit_calc(Xps1, Yps1, nonmusic_subreddits)

        with open(pickle_fn, 'w') as data_f:
            pickle.dump(topickle, data_f)

        (n_lo, n_hi, logit_1, logit_20, n_groups_gen,
                agglo_1s, agglo_20s, params, logit_params) = topickle

    ############################################
    #  Plot - subplot 1 - prediction accuracy  #
    ############################################

    plot_n_lo = 0
    plot_n_hi = 140

    snscol = sns.color_palette("Set1", n_colors=8, desat=.5)

    labelfontsize = 16
    linewidth = 2

    fig = plt.figure(figsize=(10, 4.0))
    fig.add_subplot(121)
    plt.tight_layout(pad=2, w_pad=5)
    # plt.suptitle("Feature agglomeration", size=22)
    plt.xlabel("Number of agglomerated predictors", size=labelfontsize)
    plt.ylabel("Prediction accuracy (%)", size=labelfontsize)

    plt.plot(n_groups_gen, agglo_1s, label="Agglomerated set",
             linewidth=linewidth, color=snscol[0])
    plt.plot(n_groups_gen, agglo_20s,
             label=u"Agglomerated set (≥20 subreddits)", linewidth=linewidth,
             color=snscol[1])

    plt.plot([n_lo, n_hi], [logit_1, logit_1], label="No agglomeration",
             linestyle=('dashed'), linewidth=linewidth, color=snscol[0])
    plt.plot([n_lo, n_hi], [logit_20, logit_20],
             label=u"No agglomeration (≥20 subreddits)", linestyle=('dashed'),
             linewidth=linewidth, color=snscol[1])

    axes = plt.gca()
    axes.set_xlim(plot_n_lo, plot_n_hi)
    axes.set_ylim(60, 72)

    plt.legend(fontsize=13, loc=4)

    #######################################
    #  Plot - subplot 2 - Parameter RMSD  #
    #######################################

    fig.add_subplot(122)
    plt.xlabel("Number of agglomerated predictors", size=labelfontsize)
    plt.ylabel("Mean parameter fluctuations", size=labelfontsize)

    mrmsds = []
    n_groups = []

    #var params is structured as:
    #params[k][j[i] = the ith model parameter of the jth model (jth fold in the
    #                 cross-validation) in the kth number of predictor groups
    for k, param_sets in enumerate(params):
        mrmsd = get_mrmsd(param_sets)
        mrmsds.append(mrmsd)
        n_groups.append(k+1)

    mrmsd_logit = get_mrmsd(logit_params)

    plt.plot(n_groups, mrmsds, linewidth=linewidth, color=snscol[2],
             label="Agglomerated set")
    plt.plot([n_lo, n_hi], [mrmsd_logit, mrmsd_logit],
             label="No agglomeration", linestyle=('dashed'),
             linewidth=linewidth, color=snscol[2])

    plt.legend(fontsize=13, loc=1)
    axes = plt.gca()
    axes.set_xlim(plot_n_lo, plot_n_hi)
    axes.set_ylim(0.00, 0.45)

    plt.savefig("README_figs/agglo_logit.svg")

def get_mrmsd(param_sets):
    ''' Helper function for plot_agglo_logit(). Calculates mean rmsd values for
        parameter fluctuations.
    '''

    n_params = len(param_sets[0][0])
    m_param_sets = len(param_sets)

    #Get mean
    param_mean = [0.0 for _ in range(n_params)]
    for j in range(n_params):
        for i in range(m_param_sets):
            param_mean[j] += param_sets[i][0][j]

        param_mean[j] /= m_param_sets

    #Get RMSD
    param_rmsd = [0.0 for _ in range(n_params)]
    mrmsd = 0.0

    for j in range(n_params):
        for i in range(m_param_sets):
            param_rmsd[j] += (param_sets[i][0][j] - param_mean[j])**2

        param_rmsd[j] = np.sqrt(param_rmsd[j]/m_param_sets)

    for j in range(n_params):
        mrmsd += param_rmsd[j]
    mrmsd = mrmsd/n_params
    return mrmsd

def agglo_logit_calc(Xps1, Yps1, nonmusic_subreddits):
    ''' Handles fitting and scoring of the agglomeration->logistic regression
        machine learning scheme.
    '''

    Xps1 = Xps1.toarray()

    logit = LogisticRegression()
    (n_samples_1, _) = Xps1.shape

    n_folds = 4
    rand = 0
    kf = KFold(n_samples_1, n_folds=n_folds, shuffle=True, random_state=rand)

    logit_1 = 0.0
    logit_20 = 0.0

    n_lo = 1
    n_hi = 155
    step = 1
    n_groups_gen = range(n_lo, n_hi+1, step)

    agglo_1s = [0.0 for _ in n_groups_gen]
    agglo_20s = [0.0 for _ in n_groups_gen]

    params = np.empty([len(n_groups_gen), n_folds], dtype=object)
    logit_params = []

    for i_fold, (train, test) in enumerate(kf):
        print i_fold

        logit.fit(Xps1[train], Yps1[train])
        logit_params.append(logit.coef_)

        logit_1 += (100.0*logit.score(Xps1[test], Yps1[test]))

        (Xps20_test, Yps20_test) = prune_sparse_samples(Xps1[test], Yps1[test],
                                                        threshold=20)
        (Xps20_test, Yps20_test) = balance_data(Xps20_test, Yps20_test)

        logit_20 += (100.0*logit.score(Xps20_test, Yps20_test))

        for j, n_groups in enumerate(n_groups_gen):

            agglo = phi_agglomerate(N=n_groups).fit(Xps1[train], Yps1[train])
            Xagglo_train_1, _ = agglo.transform(Xps1[train])
            Xagglo_test_1, _ = agglo.transform(Xps1[test])
            Xagglo_test_20, _ = agglo.transform(Xps20_test)

            logit.fit(Xagglo_train_1, Yps1[train])

            params[j][i_fold] = logit.coef_

            agglo_1s[j] += (100.0*logit.score(Xagglo_test_1, Yps1[test])/n_folds)
            agglo_20s[j] += (100.0*logit.score(Xagglo_test_20, Yps20_test)/n_folds)

    logit_1 /= n_folds
    logit_20 /= n_folds

    return (n_lo, n_hi, logit_1, logit_20, n_groups_gen, agglo_1s, agglo_20s,
            params, logit_params)

def graph_music_taste(Xps1, Yps1, nonmusic_subreddits, n_groups=20,
                      node_cut=2000, edge_cut=0.15):
    ''' Creates a graph of connected subreddits, colour-coded by which rank
        they come under.

        Keyword args:
        node_cut - # fans who need to post in a subreddit for it to be included
                     in visualisation
        edge_cut - # weakest edge that will be included in visualisation
    '''

    (n_samples, n_features) = Xps1.shape
    pickle_fn = "pickles/agglo_graph.pickle"

    try:
        with open(pickle_fn, 'r') as graph_file:
            g = pickle.load(graph_file)

    except IOError:
        print "No pickle of graph. Constructing."

        Xps1 = Xps1.toarray()
        (Xps1_agglo, sub_group) = phi_agglomerate(N=n_groups).\
                                            fit(Xps1, Yps1).transform(Xps1)
        coefs = LogisticRegression().fit(Xps1_agglo, Yps1).coef_[0]
        colors = get_color_rgba(coefs)

        # Create mask to only deal with subreddits above a threshold size
        sub_size = Xps1.sum(axis=0)

        # Create connections array to obtain number of users linking two arrays
        n_connections = np.zeros([n_features, n_features], dtype=int)

        for i_fan in range(n_samples):
            subs = np.nonzero(Xps1[i_fan])[0]
            for sub1, sub2 in combinations(subs, r=2):
                n_connections[sub1, sub2] += 1

        # Make vertices and assign properties
        g = Graph(directed=False)
        verts = g.add_vertex(n=n_features)
        verts = list(verts)

        sub_name = g.new_vertex_property("string")
        group = g.new_vertex_property("int")
        group_colour = g.new_vertex_property("vector<double>")
        sub_size_v = g.new_vertex_property("float")
        for i_vert in range(n_features):
            sub_name[verts[i_vert]] = nonmusic_subreddits[i_vert]
            group[verts[i_vert]] = sub_group[i_vert]
            group_colour[verts[i_vert]] = colors[sub_group[i_vert]]
            sub_size_v[verts[i_vert]] = sub_size[i_vert]

        # Make edges and assign properties
        connections = g.new_edge_property("int")
        group_av_colour = g.new_edge_property("vector<double>")
        group_av = g.new_edge_property("int")
        for a, b in combinations(range(n_features), r=2):
            e = g.add_edge(verts[a], verts[b])
            connections[e] = n_connections[a][b]
            group_av[e] = (sub_group[a]+sub_group[b])/2
            group_av_colour[e] = colors[group_av[e]]

        # Make all properties internal for pickling
        g.vertex_properties["sub_name"] = sub_name
        g.vertex_properties["sub_size"] = sub_size_v
        g.vertex_properties["group"] = group
        g.vertex_properties["group_colour"] = group_colour
        g.edge_properties["connections"] = connections
        g.edge_properties["group_av"] = group_av
        g.edge_properties["group_color"] = group_av_colour

        with open(pickle_fn, 'w') as graph_file:
            pickle.dump(g, graph_file)

    # Mask small subreddits (less than node_cut users)
    # Take log of subreddit size for size representations
    vertex_filter = g.new_vertex_property("bool")
    g.vp.sub_size_log = g.new_vertex_property("float")
    biggest = 0
    for vert in g.vertices():
        vertex_filter[vert] = g.vp.sub_size[vert] > node_cut
        g.vp.sub_size_log[vert] = np.log(g.vp.sub_size[vert])

        # Track biggest node to use as root
        if g.vp.sub_size[vert] > biggest:
            root_vert = vert
            biggest = g.vp.sub_size[vert]

    g.set_vertex_filter(vertex_filter)

    # Mask weakest edges (weight less than edge_cut)
    # Divide through # connections to make line thickness
    g.ep.line_thickness = g.new_edge_property("float")
    g.ep.line_thick_log = g.new_edge_property("float")
    edge_weight_threshold = g.new_edge_property("bool")
    for edge in g.edges():
        g.ep.line_thickness[edge] = g.ep.connections[edge]*0.003
        # g.ep.line_thick_log[edge] = np.log(g.ep.connections[edge])
        a = edge.source()
        b = edge.target()
        edge_weight = min(float(g.ep.connections[edge])/g.vp.sub_size[a],
                          float(g.ep.connections[edge])/g.vp.sub_size[b])
        edge_weight_threshold[edge] = edge_weight > edge_cut
    g.set_edge_filter(edge_weight_threshold)

    # Mask nodes with no edges (needs to converge)
    for vert in g.vertices():
        if len(list(vert.all_edges())) == 0:
            vertex_filter[vert] = False

    g.set_vertex_filter(vertex_filter)

    pos = radial_tree_layout(g, root_vert)
    graph_draw(g, pos=pos, output_size=(1000, 800),
               output="README_figs/top_subreddits_graph.svg",

               vertex_font_size=10,
               vertex_text=g.vp.sub_name,
               vertex_fill_color=g.vp.group_colour,
               vertex_size=g.vp.sub_size_log,

               edge_pen_width=g.edge_properties.line_thickness,
               edge_color=g.edge_properties.group_color)

def get_color_rgba(values, colormap=cm.bwr):
    ''' Maps a numpy.array of floats along a colour map, so that 0.0 maps to
        colormap's centre and the value furthest from 0.0 (+ or -) is mapped
        to an extreme of the colormap
    '''

    try:
        values = values.toarray()
    except AttributeError:
        pass

    N = 256.0 # colors encoded into the colormap

    magnitude = max(max(values), -min(values))
    mappings = (values*N)/(2*magnitude) + (N/2)

    rgbas = []
    for mapping in mappings.astype(int):
        rgbas.append(colormap(mapping))

    return rgbas

def get_BRBMs(Xps1, Yps1, N_range, rand, n_folds):
    ''' Trains a set of BRBMs on the dataset and saves as pickles.
        Checks if pickles already exist to not repeat work on restart.
    '''

    ##############
    #  Settings  #
    ##############

    learning_rate = 0.05
    n_iter = 1000

    ###########
    #  Train  #
    ###########

    #Unpruned samples first (Xps1, Yps1)
    (n_samples_1, n_features) = Xps1.shape
    kf = KFold(n_samples_1, n_folds=n_folds, shuffle=True, random_state=rand)
    BRBMs = np.empty([len(N_range), n_folds], dtype=object)

    for i, N in enumerate(N_range):
        for j, (train, test) in enumerate(kf):

            filename = "pickles/BRBMs/N"+str(N)+"_f"+str(j)+".pickle"
            try:
                with open(filename, 'r') as data_f:
                    BRBMs[i][j] = pickle.load(data_f)
            except IOError:
                print "Pickle not found:", filename
                print "Training..."
                rbm = BernoulliRBM(n_components=N, random_state=rand,
                                   verbose=True, n_iter=n_iter,
                                   learning_rate=learning_rate)
                rbm.fit(Xps1[train], Yps1[train])
                BRBMs[i][j] = rbm

                with open(filename, 'w') as data_f:
                    pickle.dump(rbm, data_f)

    return BRBMs

def plot_RBM(Xps1, Yps1):
    ''' Produce a plot of RBM classification accuracy and model variation
    '''

    ######################
    #  Stat/create data  #
    ######################

    n_lo = 10
    n_hi = 140

    N_range = range(n_lo, n_hi+1, 10)
    rand = 0
    n_folds = 4
    BRBMs = get_BRBMs(Xps1, Yps1, N_range, rand, n_folds)

    (n_samples_1, n_features) = Xps1.shape
    kf = KFold(n_samples_1, n_folds=n_folds, shuffle=True, random_state=rand)

    #################
    #  Test models  #
    #################

    logit = LogisticRegression()
    logit_score = [0.0 for i in N_range]
    logit_score_20 = [0.0 for i in N_range]
    logit_params = []
    params = np.empty([len(N_range), n_folds], dtype=object)
    logit_params = []

    logit_1 = 0.0
    logit_20 = 0.0

    for j_fold, (train, test) in enumerate(kf):

        (Xps20_test, Yps20_test) = prune_sparse_samples(Xps1[test], Yps1[test],
                                                        threshold=20)
        (Xps20_test, Yps20_test) = balance_data(Xps20_test, Yps20_test)

        for i, N in enumerate(N_range):

            rbm = BRBMs[i][j_fold]

            Xps1_train_trans = rbm.transform(Xps1[train])

            logit.fit(Xps1_train_trans, Yps1[train])
            params[i][j_fold] = logit.coef_

            Xps1_test_trans = rbm.transform(Xps1[test])
            logit_score[i] += 100.0*logit.score(Xps1_test_trans, Yps1[test])/n_folds

            Xps20_test_trans = rbm.transform(Xps20_test)
            logit_score_20[i] += 100.0*logit.score(Xps20_test_trans, Yps20_test)/n_folds

        logit.fit(Xps1[train], Yps1[train])
        logit_params.append(logit.coef_)
        logit_1 += (100.0*logit.score(Xps1[test], Yps1[test]))/n_folds
        logit_20 += (100.0*logit.score(Xps20_test, Yps20_test))/n_folds

    ############################################
    #  Plot - subplot 1 - prediction accuracy  #
    ############################################

    plot_n_lo = 0
    plot_n_hi = n_hi

    snscol = sns.color_palette("Set1", n_colors=8, desat=.5)

    labelfontsize = 16
    linewidth = 2

    fig = plt.figure(figsize=(10, 4.0))
    fig.add_subplot(121)
    plt.tight_layout(pad=2, w_pad=5)
    # plt.suptitle("Restricted Boltzmann Machines", size=22)
    plt.xlabel("Number of hidden units", size=labelfontsize)
    plt.ylabel("Prediction accuracy (%)", size=labelfontsize)

    plt.plot(N_range, logit_score, label="RBM features",
             linewidth=linewidth, color=snscol[0])
    plt.plot(N_range, logit_score_20, label=u"RBM features (≥20 subreddits)",
             linewidth=linewidth, color=snscol[1])

    plt.plot([plot_n_lo, plot_n_hi], [logit_1, logit_1], label="No RBM",
             linestyle=('dashed'), linewidth=linewidth, color=snscol[0])
    plt.plot([plot_n_lo, plot_n_hi], [logit_20, logit_20],
             label=u"No RBM (≥20 subreddits)", linestyle=('dashed'),
             linewidth=linewidth, color=snscol[1])

    axes = plt.gca()
    axes.set_xlim(plot_n_lo, plot_n_hi)
    axes.set_ylim(60, 72)

    plt.legend(fontsize=12.5, loc=4)

    #######################################
    #  Plot - subplot 2 - Parameter RMSD  #
    #######################################

    fig.add_subplot(122)
    plt.xlabel("Number of hidden units", size=labelfontsize)
    plt.ylabel("Mean parameter fluctuations", size=labelfontsize)

    mrmsds = []

    #var params is structured as:
    #params[k][j[i] = the ith model parameter of the jth model (jth fold in the
    #                 cross-validation) in the kth number of predictor groups
    for k, param_sets in enumerate(params):
        mrmsd = get_mrmsd(param_sets)
        mrmsds.append(mrmsd)

    mrmsd_logit = get_mrmsd(logit_params)

    plt.plot(N_range, mrmsds, linewidth=linewidth, color=snscol[2],
             label="RBM features")
    plt.plot([plot_n_lo, plot_n_hi], [mrmsd_logit, mrmsd_logit],
             label="No RBM", linestyle=('dashed'),
             linewidth=linewidth, color=snscol[2])

    plt.legend(fontsize=12.5, loc=4)
    axes = plt.gca()
    axes.set_xlim(plot_n_lo, plot_n_hi)
    axes.set_ylim(0.00, 0.45)

    plt.savefig("README_figs/RBMs_logit.svg")
