import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.model_selection import train_test_split
import warnings
from Clients import Client, Clients
from Recommender import ColabRecommender, ContentRecommender

warnings.filterwarnings("ignore")

data          = pd.read_csv('./data/german_credit_data.csv')
numeric_data  = data.loc[:,["Age","Credit amount","Duration"]]
clients       = Clients(numeric_data)

X = clients.getX()
y = clients.getY()


X_train, X_test, Y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1, stratify=y)
contentRecommender = ContentRecommender(X_train,Y_train,10)
# client             = Client(age=45,credit_amount=2000,duration=2)
contentRecommender.knn_fit()

print(contentRecommender.recommend(X_test)[0:100])

