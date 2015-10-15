#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sqlite3
from subreddits import subreddit_genres, genres, music_subreddits
import numpy as np
from scipy.sparse import coo_matrix

"""
Functions to produce an (X,Y) dataset where the samples are reddit users, the
outcomes are the category of 'fan' the user is (one of classical, hiphop, 
rockmetal, or electronic) and the predictors are which offtopic subreddits the
user has posted in.
"""

# TODO: Potential improvements
# - Cleverer offtopic subreddit selection in get_offtopic_subreddits
# - SQL enhancements. e.g. creating a smaller temp table.

def get_dataset_SQL(db_file, table_name, fans_limit=50000, offtopic_limit=2000):
    """
    Driver to return the (X,Y) dataset described in the module docstring.

    Arguments:

    db_file -- location of SQL database file.
    table_name -- name of main table to read from.
    """

    conn = sqlite3.connect(db_file)
    conn.text_factory = str
    c = conn.cursor()

    # Create sparse array of users, categorised by which genre of music
    # subreddit they contribute to (Y), and which unrelated subreddits they
    # contribute to (X)
    print ( "Selecting fans of each genre." )
    fans_by_genre = get_fans_by_genre(c, table_name, subreddit_genres, LIMIT=fans_limit)

    # Create flat list of all fans and Y array of outcomes
    # Guarentee synchronicity by creating together
    fans_and_genres = [(i_genre, fans_by_genre[genre][j])
                        for i_genre, genre in enumerate(genres)
                        for j in range(len(fans_by_genre[genre]))]
    Y, all_fans = zip(*fans_and_genres)
    Y = np.array(Y)

    for (key, val) in fans_by_genre.items():
        print ( "{0:d} fans of {1:s}".format(len(val), key) )
    print ( "" )

    print ( "Populating list of unrelated subreddits used by music fans." )
    nonmusic_subreddits = get_offtopic_subreddits(c, table_name, all_fans, music_subreddits, LIMIT=offtopic_limit)
    print ( "" )

    print ( "Creating sparse array of predictors for each user." )
    n_samples = len(all_fans)
    X = get_array_user_subreddits(c, table_name, all_fans, nonmusic_subreddits)
    print ( "" )

    conn.close()

    return (X, Y, nonmusic_subreddits)

#################################
#  Populate dict of genre fans  #
#################################

def get_fans_by_genre(c, table_name, subreddit_genres, LIMIT=10000):
    '''
    Returns: dict where keys are genres and values are authors contributing to
             subreddits of that genre and none of the other genres.
    '''

    fans_by_genre = {}

    #Select distinct authors from database
    for genre in genres:
        c.execute (
            "SELECT DISTINCT author\
             FROM {}\
             WHERE subreddit IN ( \"{}\" )\
             LIMIT {}\
             ".format(table_name, '", "'.join(subreddit_genres[genre]), LIMIT)
        )

        fans_by_genre[genre] = c.fetchall()
        fans_by_genre[genre] = [x[0] for x in fans_by_genre[genre]]
    print ("")

    #Erase authors who post about more than one genre
    #This also automates removal of "[deleted]" and many bots
    duplicate_authors = []
    removals = 0

    for i, genre1 in enumerate(genres):
        for genre2 in genres[i+1:]:
            duplicate_authors += set(fans_by_genre[genre2]).intersection(fans_by_genre[genre1])

    for genre in fans_by_genre:
        for duplicate_author in duplicate_authors:
            try:
                fans_by_genre[genre].remove(duplicate_author)
                removals += 1
            except ValueError:
                pass
    return fans_by_genre

#####################################################################
#  Get list of all offtopic subreddits authors have contributed to. #
#####################################################################

def get_offtopic_subreddits(c, table_name, all_fans, excluded_subreddits, LIMIT=10000):
    '''
    Returns list of "offtopic" subreddits users have contributed to.
    '''

    # Retrieves up to LIMIT of the most popular subreddits.
    # Inexact, because it counts posts, not number of unique posters.
    # Potential for improvement by seeking subreddits whose popularity
    # most divides fans.

    c.execute (
        "SELECT DISTINCT subreddit "\
        "FROM {} "\
        "WHERE author IN ( \"{}\" ) "\
        "AND subreddit NOT IN ( \"{}\" ) "\
        "GROUP BY subreddit "\
        "ORDER BY count(subreddit) DESC "\
        "LIMIT {}"\
        .format(table_name,
                 '", "'.join( all_fans ),
                 '", "'.join( excluded_subreddits ),
                 LIMIT
                 )
     )

    offtopic_subreddits = [subreddit[0] for subreddit in c.fetchall()]
    # Python will use this to do a binary search:
    offtopic_subreddits.sort()

    return offtopic_subreddits

##############################################################
#  Populate sparse array of each user's offtopic subreddits  #
##############################################################

def get_array_user_subreddits(c, table_name, all_fans, offtopic_subreddits):
    n_samples = len(all_fans)
    n_features = len(offtopic_subreddits)
    i_fans = []
    j_subreddits = []

    for i_fan, fan in enumerate(all_fans):
        # print (
        c.execute (
            "SELECT DISTINCT subreddit "\
            "FROM {} "\
            "WHERE author = \"{}\" "\
            "AND subreddit IN ( \"{}\" ) "\
            .format( table_name, fan, '", "'.join(offtopic_subreddits ) )
        )
        fan_subreddits = [subreddit[0] for subreddit in c.fetchall()]

        for subreddit in fan_subreddits:
            i_fans.append(i_fan)
            j_subreddits.append( offtopic_subreddits.index(subreddit) )

    values = np.ones ([ len(i_fans) ], dtype=bool)
    X = coo_matrix( ( values, (i_fans, j_subreddits) ) )

    return X

