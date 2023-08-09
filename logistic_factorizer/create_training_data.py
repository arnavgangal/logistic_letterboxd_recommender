#!/usr/local/bin/python3.11

import pandas as pd
import pickle
import pymongo
from db_connect import connect_to_db


def get_sample(cursor, iteration_size):
    """ Get a sample of fixed size from a MongoDB collection

    :param cursor: Specific collection in the MongoDB database
    :param iteration_size: Number of items in the sample, int
    :return: list of samples from the collection
    """
    # Get the samples, return them as a list
    while True:
        try:
            rating_sample = cursor.aggregate([
                {"$sample": {"size": iteration_size}}
            ])
            return list(rating_sample)
        except pymongo.errors.OperationFailure:
            print("Encountered $sample operation error. Retrying...")


def create_training_data(db_client, sample_size=200000):
    """ Creates a random user-item-rating dataframe of specified size from the database

    :param db_client: MongoDB client
    :param sample_size: int, total number of reviews to put into data frame
    :return: Training data frame with sample_size number of rows
    """
    # Get the ratings collection
    ratings = db_client.ratings

    # Empty list, and tracker for number of unique ratings
    all_ratings = []
    unique_records = 0
    # Until we reach the sample size
    while unique_records < sample_size:
        # Get a sample of 10000 from the ratings collection as a list
        rating_sample = get_sample(ratings, 10000)
        # Add the sample to all the ratings
        all_ratings += rating_sample
        # Get number of unique records (i.e. ignore duplicate reviews)
        unique_records = len(set([(x['movie_id'] + x['user_id']) for x in all_ratings]))

    # Create a dataframe of collected ratings
    df = pd.DataFrame(all_ratings)
    # Drop the object id column, all duplicates, and keep the top sample_size rows
    df = df[["user_id", "movie_id", "rating_val"]]
    df.drop_duplicates(inplace=True)
    df = df.head(sample_size)

    print(df.head())

    return df


def create_movie_data_sample(db_client, movie_list):
    """ Create a dataframe of movie information

    :param db_client: MongoDB client
    :param movie_list: Movies to keep in the data frame
    :return: (data frame) of movie info
    """
    # Connect to collection
    movies = db_client.movies
    # Filter for movies that we want to include
    included_movies = movies.find({"movie_id": {"$in": movie_list}})

    # Create a dataframe of included movies
    movie_df = pd.DataFrame(list(included_movies))
    # Keep only relevant columns
    movie_df = movie_df[['movie_id', 'image_url', 'movie_title', 'year_released']]
    # Pull the image urls from website
    movie_df['image_url'] = movie_df['image_url'].fillna('').str.replace('https://a.ltrbxd.com/resized/', '',
                                                                         regex=False)
    movie_df['image_url'] = movie_df['image_url'].fillna('').str.replace(
        'https://s.ltrbxd.com/static/img/empty-poster-230.c6baa486.png', '', regex=False)

    return movie_df


if __name__ == "__main__":
    # Set variables
    min_review_threshold = 15

    # Connect to MongoDB client
    db_name, client, tmdb_key = connect_to_db()
    db = client[db_name]

    # Generate training data sample, 200000 unique reviews
    training_df = create_training_data(db, 200000)
    print(training_df.shape)

    # Create review counts dataframe
    # Filter out movies with fewer than min_threshold reviews
    review_count = db.ratings.aggregate([
        {"$group": {"_id": "$movie_id", "review_count": {"$sum": 1}}},
        {"$match": {"review_count": {"$gte": min_review_threshold}}}
    ])

    # Convert review counts into a dataframe
    review_counts_df = pd.DataFrame(list(review_count))
    review_counts_df.rename(columns={"_id": "movie_id", "review_count": "count"}, inplace=True)

    # Get all movie ids as a list
    threshold_movie_list = review_counts_df['movie_id'].to_list()

    # Generate movie data CSV from movie IDs
    movie_df = create_movie_data_sample(db, threshold_movie_list)
    print(movie_df.head())
    print(movie_df.shape)

    # Use movie_df to remove any items from threshold_list that do not have a "year_released"
    # This means it's a collection of more popular movies (e.g. movie trilogies), so don't include it
    retain_list = movie_df.loc[(movie_df['year_released'].notna() & movie_df['year_released'] != 0.0)][
        'movie_id'].to_list()

    # Overlap of movies that are over the threshold, and have a year released
    threshold_movie_list = [x for x in threshold_movie_list if x in retain_list]

    # Store Data (write as binary)
    with open('models/threshold_movie_list.txt', 'wb') as fp:
        pickle.dump(threshold_movie_list, fp)

    # Save user-item-rating df as a cvs
    training_df.to_csv('data/training_data.csv', index=False)
    # Save review counts as a csv
    review_counts_df.to_csv('data/review_counts.csv', index=False)
    # Save movie metadata as a csv
    movie_df.to_csv('../static/data/movie_data.csv', index=False)
