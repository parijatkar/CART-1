# -*- coding: utf-8 -*-
"""
Created on Wed Jan 1 14:57:08 2020

@author: parij
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn import tree
import graphviz
from sklearn.tree import export_text
import pylab

col = [ 'subjectid','age','height','weight','size1','size2','pcount',
       'bfcount', 'fhistory','positive']

dtype1 = ['int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int']
df = pd.read_csv('C:\\Users\\parij\\CART-1\\BreastCancerSocialData.csv', header=None, 
                 names=col,sep=',', dtype=np.int32)


df.head()

df.info()

sns.countplot(df['positive'])


X = df.drop(['positive', 'subjectid'], axis=1)
y = df[['positive']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state=42)


clf_model = DecisionTreeClassifier(criterion="gini", random_state=42,max_depth=4, min_samples_leaf=5)   
clf_model.fit(X_train,y_train)

DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)

y_predict = clf_model.predict(X_test)


Z = accuracy_score(y_test,y_predict)
print('Accuracy:', Z)

#CM = confusion_matrix(y_train, y_test)
#print(CM)

target = list(df['positive'].unique())
map_target = np.asarray(target).astype(str)
print(map_target)

print(target)
feature_names = list(X.columns)

print(feature_names)


dot_data = tree.export_graphviz(clf_model,
                                out_file=None, 
                      feature_names=feature_names,  
                      class_names=map_target,  
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  

#graph.view()

filename = graph.render(filename='DecisionTreeGraph')

pylab.savefig('filename.png')


r = export_text(clf_model, feature_names=feature_names)
print(r)