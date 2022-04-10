#importing modules 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

#Loading the data

data=pd.read_csv("./Mall_Customers.csv")
X = data.iloc[:, :].values
data.head()


dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()

#Train the model

clustering = AgglomerativeClustering(n_clusters = 5).fit(X)
y_hc = clustering.fit_predict(X)


#Visualizing cluster 

plt.scatter(X[y_hc == 0 , 0] ,X[y_hc == 0 , 1], c = 'red' , label = 'Cluster 1' )
plt.scatter(X[y_hc == 1 , 0] ,X[y_hc == 1 , 1], c = 'green' , label = 'Cluster 2' )
plt.scatter(X[y_hc == 2 , 0] ,X[y_hc == 2 , 1], c = 'pink' , label = 'Cluster 3' )
plt.scatter(X[y_hc == 3 , 0] ,X[y_hc == 3 , 1], c = 'blue' , label = 'Cluster 4' )
plt.scatter(X[y_hc == 4 , 0] ,X[y_hc == 4 , 1], c = 'orange' , label = 'Cluster 5' )
plt.title("Cluster of Customers")
plt.xlabel("Annual Income(K$)")
plt.ylabel("Spending Score(1-100)")
plt.legend()
plt.show()