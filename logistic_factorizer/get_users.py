# Imports
from pymongo.operations import ReplaceOne, UpdateOne
import requests
from bs4 import BeautifulSoup
import pymongo
from pymongo.errors import BulkWriteError
from pprint import pprint
from tqdm import tqdm
from db_connect import connect_to_db

# Connect to MongoDB client
db_name, client, tmdb_key = connect_to_db()
db = client[db_name]
users = db.users

# URL format to scrape for
base_url = "https://letterboxd.com/members/popular/this/week/page/{}/"

# Number of pages to scrape
total_pages = 6

# Progress bar
pbar = tqdm(range(1, total_pages + 1))

# Scraping loop
for page in pbar:
    pbar.set_description(f"Scraping page {page} of {total_pages} of top users")

    # HTTP Get request
    r = requests.get(base_url.format(page))
    # Parse with BeautifulSoup - find a table, returns a list of matching elements
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", attrs={"class": "person-table"})
    rows = table.findAll("td", attrs={"class": "table-person"})

    # Iterate over all users on the page
    update_operations = []
    for row in rows:
        # Get link to the user page
        link = row.find("a")["href"]
        # Get the username, and number of reviews
        username = link.strip('/')
        display_name = row.find("a", attrs={"class": "name"}).text.strip()
        num_reviews = int(row.find("small").find("a").text.replace('\xa0', ' ').split()[0].replace(',', ''))

        # Create a dictionary with user info to write to the database
        user = {
            "username": username,
            "display_name": display_name,
            "num_reviews": num_reviews
        }

        # Append to the list of update operations
        # Filter documents for the provided username
        # Write the user information to the doc in the db
        # Create new doc if it does not exist
        update_operations.append(
            UpdateOne({
                "username": user["username"]
            },
                {"$set": user},
                upsert=True
            )
        )

        # users.update_one({"username": user["username"]}, {"$set": user}, upsert=True)

    # Bulk write to the database
    try:
        if len(update_operations) > 0:
            users.bulk_write(update_operations, ordered=False)
    except BulkWriteError as bwe:
        print(bwe.details)