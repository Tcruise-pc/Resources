import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
dataset = pd.read_csv("Wine.csv")

X = dataset.iloc[: ,:-1].values

y = dataset.iloc[: , -1].values

#split the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Apply LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

#Visualising Training Set Results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(
    np.arange(start = X_set[: , 0].min()-1 , stop = X_set[: , 0].max() +1 , step = 0.25),
    np.arange(start = X_set[: , 1].min()-1 , stop = X_set[: , 1].max() +1 , step = 0.25),
    )
plt.contourf(X1, X2, lr.predict(np.array([X1.ravel() , X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red' ,'blue', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j,1],
               c = ListedColormap(('red', 'blue', 'green'))(i), label = j)
    
plt.title("Training Set")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend()
plt.show()

#Visualising Tes Set Results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(
    np.arange(start = X_set[: , 0].min()-1 , stop = X_set[: , 0].max() +1 , step = 0.25),
    np.arange(start = X_set[: , 1].min()-1 , stop = X_set[: , 1].max() +1 , step = 0.25),
    )
plt.contourf(X1, X2, lr.predict(np.array([X1.ravel() , X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red' ,'blue', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j,1],
               c = ListedColormap(('red', 'blue', 'green'))(i), label = j)
    
plt.title("Test Set")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend()
plt.show()

