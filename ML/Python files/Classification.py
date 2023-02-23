import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import pickle


PATH = "./clusters.csv"


def open_X_y(path: str, class_name : str):
    """
    """
    # opening the dataset
    churn = pd.read_csv(path)
    churn.drop(['Unnamed: 0'], axis=1, inplace=True)
    # selecting only important features
    if class_name == 'cluster':
        churn = churn[['Total_Trans_Ct', 'Total_Revolving_Bal', 'Total_Relationship_Count',
                    'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Ct_Chng_Q4_Q1', 'Avg_Open_To_Buy', 'Customer_Age', 'cluster']]
    else : 
        churn = churn[['Attrition_Flag','Total_Trans_Ct', 'Total_Revolving_Bal', 'Total_Relationship_Count',
                    'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Ct_Chng_Q4_Q1', 'Avg_Open_To_Buy', 'Customer_Age']]
        
    churn = churn.dropna()
    less = churn.reindex(sorted(churn.columns), axis=1)

    #divide to feature and results
    X = less.drop(class_name,axis=1).values
    y = less[class_name].values
 
    return X,y

def split(X: np.array, y: np.array):
    """
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    return X_train, X_test, y_train, y_test


def sampling(X: np.array, y: np.array):
    """
    """
    ros = RandomOverSampler(sampling_strategy="not majority")
    X_ros, y_ros = ros.fit_resample(X,y)

    return X_ros,y_ros

def model(X : np.array,y : np.array,path_model : str):
    """
    """
    X_train, X_test, y_train, y_test = split(X,y)

    X_train, y_train = sampling(X_train,y_train)

    steps=[("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(random_state=45, max_depth=7,criterion='gini', max_features=None))]

    pipe = Pipeline(steps)

    pipe.fit(X_train,y_train)
    # save as pickle
    pickle.dump(pipe, open( path_model, 'wb'))
    # print score
    y_pred = pipe.predict(X_test)
    print(pipe.score(X_test,y_test))
    #print(classification_report(y_test, y_pred))
    # test of prediction
    new_X = (np.array([200,777,10,1.0,1000,5,100,20])).reshape(-1, 8)
    print(pipe.predict(new_X))
    print(pipe.predict_proba(new_X))

    return pipe

chrun_model = '../Models/Churn_class.pkl'
cluster_model = '../Models/Clusters_Class.pkl'

X,y = open_X_y(PATH, 'Attrition_Flag')
model(X,y, chrun_model)
X_cluster, y_cluster = open_X_y(PATH, 'cluster')
model(X_cluster, y_cluster, cluster_model)