import pandas as pd
import pickle
import random
import numpy as np

# SK-learn module for recommender systems
from surprise import Dataset
from surprise import Reader
from surprise.dump import dump

from logistic_factorizer.LogFactorizer import LogFactorizer


def build_model(df, user_data):
    """ Builds and returns a logistic factorizer model trained with the user included

    :param df: a training pandas dataframe in the user-item-rating format
    :param user_data: watch data from a specific user
    :return: (A trained logistic factorizer model, list of movies that the user has already watched)
    :rtype: (LogFactorizer, list of strings)
    """

    # Set random seed so that returned recs are always the same for same user with same ratings
    # This might make sense so that results are consistent, or you might want to refresh with different results
    my_seed = 12
    random.seed(my_seed)
    np.random.seed(my_seed)

    # Get all watch data for the user, turn into df
    user_rated = [x for x in user_data if x['rating_val'] > 0]
    user_df = pd.DataFrame(user_rated)

    # Vertically conjoin training and user data frames
    df = pd.concat([df, user_df]).reset_index(drop=True)
    df.drop_duplicates(inplace=True)
    del user_df

    # Build a Reader object to parse file containing ratings
    reader = Reader(rating_scale=(0, 1))

    # Create Dataset object from dataframe
    data = Dataset.load_from_df(df[["user_id", "movie_id", "rating_val"]], reader)
    del df

    # Initialize LogFactorizer algorithm
    # Default is 50 epochs
    algo = LogFactorizer(n_factors=10, n_epochs=30, lr_users=0.0001, lr_items=0.01, verbose=True)

    # Create a Trainset to fit the logistic matrix factorizer
    training_set = data.build_full_trainset()
    algo.fit(training_set)

    # Gets a list of the movies that the user has already watched
    user_watched_list = [x['movie_id'] for x in user_data]

    return algo, user_watched_list


if __name__ == "__main__":
    import os
    if os.getcwd().endswith("logistic_factorizer"):
        from get_user_ratings import get_user_data
    else:
        from logistic_factorizer.get_user_ratings import get_user_data

    # Load user-item-rating training data
    df = pd.read_csv('data/training_data.csv')

    # Get user data (specific user)
    user_data = get_user_data("fumilayo")[0]

    # Train the recommender, get watched movies from a given user
    algo, user_watched_list = build_model(df, user_data)

    # Pickle wrapper to dump the algorithm (no predictions)
    dump("models/mini_model.pkl", predictions=None, algo=algo, verbose=1)

    # Dump the user watched list
    with open("models/user_watched.txt", "wb") as fp:
        pickle.dump(user_watched_list, fp)