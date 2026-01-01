from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client.recipe_database
recipes_collection = db.recipes
users_collection = db.users

def fetch_recipe_details(recipe_name):
    recipe = recipes_collection.find_one({'name': recipe_name})
    return recipe or {'name': 'Unknown', 'ingredients': 'N/A', 'recipe_steps': 'N/A', 'calories': 'N/A', 'nutrients': 'N/A'}

def insert_user(username, password):
    if users_collection.find_one({'username': username}):
        return False
    users_collection.insert_one({'username': username, 'password': password})
    return True

def authenticate_user(username, password):
    user = users_collection.find_one({'username': username, 'password': password})
    return user is not None
