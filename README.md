# A Logistic Matrix Factorization Method for Letterboxd Recommendation

The code in this repository is based on Sam Learner's Letterboxd recommender, and uses much of the same database setup and webscraping, without the Redis frontend. To run the code yourself locally, setup the MongoDB instance by:
1. Starting up a local MongoDB server on the default port (27017)
2. Add a `db_config` file to the `logistic_factorizer` directory, with a config variable such as: `config = { 'MONGO_DB': 'letterboxd-implicit', 'CONNECTION_URL': 'mongodb://localhost:27017/'}`, and optionally a TMDB API key as `tmdb_key = "EXAMPLEKEY1234"`.
   
You can then run the following files in order:
1. `get_users.py`
2. `get_ratings.py`
3. `get_movies.py`
   
to build the database. Note that the first scrape may take several hours, but subsequent scrapes will be faster. The initial scrape can be sped up by using `lxml` instead of the `html.parser`. To build and run the model, run the following files:
1. `create_training_data.py`
2. `build_model.py`
3. `run_model.py`
making sure to change the username supplied to the `get_user_data` function in `build_model.py`, and the `run_model` function in `run_model.py`. You may find it helpful to tweak hyperparameters in the `build_model` file, to adjust the performance of your model. The default values in the constructor of the `LogFactorizer` class were ones that I experimentally found to be effective. 
