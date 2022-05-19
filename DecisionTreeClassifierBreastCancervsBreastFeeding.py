# -*- coding: utf-8 -*-
"""
Created on Thu May 19 14:57:08 2022

@author: parij
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

col = [ 'subjectid','age','height','weight','size1','size2','pcount',
       'bfcount',	'fhistory',	'positive']

df = pd.read_csv('C:\\Users\\parij\\CART-1\\BreastCancerSocialData.csv',names=col,sep=',')
df.head()

df.info()

sns.countplot(df['positive'])

from sklearn.model_selection import train_test_split
X = df.drop(['positive', 'subjectid'],axis=1)
y = df[['positive']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state=42)

from sklearn.tree import DecisionTreeClassifier
clf_model = DecisionTreeClassifier(criterion="gini", random_state=42,max_depth=3, min_samples_leaf=5)   
clf_model.fit(X_train,y_train)

DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)

y_predict = clf_model.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
accuracy_score(y_test,y_predict)

target = list(df['positive'].unique())
feature_names = list(X.columns)

from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(clf_model,
                                out_file=None, 
                      feature_names=feature_names,  
                      class_names=target,  
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  

graph

from sklearn.tree import export_text
r = export_text(clf_model, feature_names=feature_names)
print(r)