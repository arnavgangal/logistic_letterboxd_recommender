from surprise.dump import load
import pickle
import random
import os
import pymongo

try:
    from db_config import config
except ImportError:
    config = None


def get_top_n(predictions, n=20):
    """ Get the top n predictions from some predictions

    :param predictions: predictions
    :param n: number of predictions to get
    :return: list of top n predictions
    """
    top_n = [(iid, est) for uid, iid, true_r, est, _ in predictions]
    top_n.sort(key=lambda x: (x[1], random.random()), reverse=True)

    return top_n[:n]


def run_model(username, algo, user_watched_list, threshold_movie_list, num_recommendations=20):
    """ Runs the provided algorithm on the given user, returns recs

    :param username: str, username to recommend for
    :param algo: Algorithm, inherits from AlgoBase
    :param user_watched_list: List of watched movies, as dicts
    :param threshold_movie_list: List of movies with enough reviews, as dicts
    :param num_recommendations: int, number of recommendations
    :return: List of dicts, containing recs
    """
    # Connect to MongoDB Client
    if config:
        db_name = config["MONGO_DB"]
    else:
        db_name = os.environ.get('MONGO_DB', '')

    serverless_connection = True
    if config:
        if config["CONNECTION_URL"]:
            connection_url = config["CONNECTION_URL"]
        else:
            connection_url = f'mongodb+srv://{config["MONGO_USERNAME"]}:{config["MONGO_PASSWORD"]}@cluster0.{config["MONGO_CLUSTER_ID"]}.mongodb.net/{db_name}?retryWrites=true&w=majority'
            serverless_connection = False
    else:
        connection_url = os.environ.get('CONNECTION_URL', '')

    if serverless_connection:
        # client = pymongo.MongoClient(connection_url, server_api=pymongo.server_api.ServerApi('1'))
        client = pymongo.MongoClient(connection_url)
    else:
        client = pymongo.MongoClient(connection_url)

    # Connect to database
    db = client[db_name]

    # Get a list of unwatched movies with enough reviews
    unwatched_movies = [x for x in threshold_movie_list if x not in user_watched_list]

    # Create a list of tuples with info to be filled in for predictions
    prediction_set = [(username, x, 0) for x in unwatched_movies]

    # Run the algorithm on the unwatched movies
    predictions = algo.test(prediction_set)

    # Get the top n predictions
    top_n = get_top_n(predictions, num_recommendations)

    # Set info about the movies
    # movie_fields = ["image_url", "movie_id", "movie_title", "year_released", "genres", "original_language",
    #                 "popularity", "runtime", "release_date"]
    movie_fields = ["movie_title", "year_released", "genres"]
    movie_data = {x["movie_id"]: {k: v for k, v in x.items() if k in movie_fields} for x in
                  db.movies.find({"movie_id": {"$in": [x[0] for x in top_n]}})}

    # Construct the return object with movie info
    return_object = [{"movie_id": x[0], "predicted_rating": round(x[1], 3),
                      "movie_data": movie_data[x[0]]} for x in top_n]

    # Sort the objects by unclipped rating, return
    return_object.sort(key=lambda x: (x["predicted_rating"]), reverse=True)
    return return_object


if __name__ == "__main__":
    # Load in the movies the user has watched
    with open("models/user_watched.txt", "rb") as fp:
        user_watched_list = pickle.load(fp)

    # Load in all movies that have at least threshold number of reviews
    with open("models/threshold_movie_list.txt", "rb") as fp:
        threshold_movie_list = pickle.load(fp)

    # Load in the algorithm
    algo = load("models/mini_model.pkl")[1]

    # Run the model on the specified user, and print
    recs = run_model("fumilayo", algo, user_watched_list, threshold_movie_list, 25)
    for rec in recs:
        print(rec["movie_data"]["movie_title"])
