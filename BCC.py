# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 22:22:58 2020

@author: Rushi
"""
import pandas as pd
import numpy as np

data=pd.read_csv('data.csv')

data.columns
data.dtypes

x=data.iloc[:,2:-1]
data['diagnosis'] = np.where(data['diagnosis'] == 'M',1,0)
y=data['diagnosis']

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(data.corr(),annot=True)
plt.show()

from sklearn.model_selection import train_test_split,KFold,cross_val_score
X_train, X_test, Y_train, Y_test = train_test_split (x, y, test_size = 0.25, random_state=0)

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

models=[('DecisionTree', DecisionTreeClassifier()),('KNN', KNeighborsClassifier()),('SVM', SVC())]

for name,model in models:
    kfold=KFold()
    cv_result=cross_val_score(model, X_train,Y_train,cv=kfold,scoring='accuracy')
    print(name,cv_result.mean())

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



pipeln=(('ScaledDT',Pipeline([('Scaler', StandardScaler()),('BT',DecisionTreeClassifier())])),('ScaledKNN',Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsClassifier())])),('ScaledSVM',Pipeline([('Scaler', StandardScaler()),('SVM',SVC())])))
for name,model in pipeln:
    kfold=KFold()
    cv_result=cross_val_score(model, X_train,Y_train,cv=kfold,scoring='accuracy')
    print(name,cv_result.mean())
    

scaler = StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

classifier = SVC(kernel = 'linear', random_state = 0)

from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.01,0.1,1,10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, Y_train)


accuracy = grid_search.best_score_
print(accuracy)
print(grid_search.best_params_)

classifier = SVC(C=0.1,kernel = 'linear', random_state = 0)
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(Y_test,
                        Y_pred)
print(accuracy)
import pickle
pickle.dump(classifier, open('model.pkl','wb'))