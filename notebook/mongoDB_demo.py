import pandas as pd
import pymongo

df = pd.read_csv('notebook\data.csv')
df.head()


# df should be converted into dict before we push it to mongodb
data = df.to_dict(orient='records')
# data


DB_NAME = "Proj1"
COLLECTION_NAME = "Proj1-Data"
CONNECTION_URL = "mongodb+srv://telrmanoj_db_user:1NUkrKQo6MS5j2GY@cluster0.pq0rllk.mongodb.net/?appName=Cluster0"
# above, either remove your credentials or delete the mongoDB resource bofore pushing it to github.


client = pymongo.MongoClient(CONNECTION_URL)
data_base = client[DB_NAME]
collection = data_base[COLLECTION_NAME]


# Uploading data to MongoDB
rec = collection.insert_many(data)