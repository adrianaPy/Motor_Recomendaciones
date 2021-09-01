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
indices = np.arange(1000)

#An'alisis de datos...
X = clients.getX()
y = clients.getY()


X_train, X_test, Y_train, y_test,indices_train,indices_test  = train_test_split(X,y,indices,test_size=0.2, random_state=1, stratify=y)
contentRecommender = ContentRecommender(X_train,Y_train)

# client             = Client(age=45,credit_amount=2000,duration=2)
# print(indices_test)




contentRecommender.set_neighbors(15)
contentRecommender.knc_fit()
clients_test =  data.loc[indices_test,["Age","Credit amount","Duration"]]
clients_test['Key'] = contentRecommender.knc_recommend(X_test)[0:200]
print(clients_test)
