import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.model_selection import train_test_split
import warnings
from Clients import Client, Clients
from Recommender import ColabRecommender
from numpy import random

warnings.filterwarnings("ignore")

data          = pd.read_csv('./data/german_credit_data.csv')
numeric_data  = data.loc[:,["Age","Credit amount","Duration"]]
clients       = Clients(numeric_data)

X = clients.getX()
y = clients.getY()


data['Key'] =  random.randint(10,size=(1000))
data['Cal_A'] = random.randint(5,size=(1000))
data['Cal_B'] = random.randint(5,size=(1000))
data['Cal_C'] = random.randint(5,size=(1000))
data['Cal_D'] = random.randint(5,size=(1000))
data['Cal_E'] = random.randint(5,size=(1000))
data['Cal_F'] = random.randint(5,size=(1000))
data['Cal_G'] = random.randint(5,size=(1000))
data['Cal_H'] = random.randint(5,size=(1000))
data['Cal_I'] = random.randint(5,size=(1000))
data['Cal_J'] = random.randint(5,size=(1000))


client_product_matrix = data.iloc[:,10:]
# print(client_product_matrix)


y = client_product_matrix['Key'].values
X = client_product_matrix.drop(columns=['Key'])


#Clasifier
X_train, X_test, Y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1, stratify=y)
colabRecommender = ColabRecommender(X_train,Y_train)
colabRecommender.set_neighbors(10)
colabRecommender.knc_fit()
print(colabRecommender.recommend(X_test)[0:200])


#Nearest neighbors
colabRecommender_NN = ColabRecommender(X_train=X,Y_train=Y_train)
colabRecommender_NN.set_neighbors(10)
colabRecommender.nn_fit()
d,i = colabRecommender.knn_kneighbors()
print(d,i)
