import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import pickle


PATH = "./Cleaned_ML_Unknonw.csv"


def open_X_y(path: str):
    """
    """
    # opening the dataset
    churn = pd.read_csv(path)
    churn.drop(['Unnamed: 0'], axis=1, inplace=True)
    # selecting only important features
    churn = churn[['Attrition_Flag', 'Total_Trans_Ct', 'Total_Revolving_Bal', 'Total_Relationship_Count',
                  'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Ct_Chng_Q4_Q1', 'Avg_Open_To_Buy', 'Customer_Age']]
    less = churn.reindex(sorted(churn.columns), axis=1)
    print(less.columns)
    #divide to feature and results
    X = less.drop('Attrition_Flag',axis=1).values
    y = less['Attrition_Flag'].values
    print('opened')
    return X,y

def sampling(X: np.array, y: np.array):
    """
    """
    ros = RandomOverSampler(sampling_strategy="not majority")
    X_ros, y_ros = ros.fit_resample(X,y)
    print('sampling')
    return X_ros,y_ros

def model(X : np.array,y : np.array):
    """
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    steps=[("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(random_state=45, max_depth=7,criterion='gini', max_features=None))]

    pipe = Pipeline(steps)
    print('pipe step')
    pipe.fit(X_train,y_train)
    # save as pickle
    pickle.dump(pipe, open('../Models/Tree_Class.pkl', 'wb'))
    #print score
    print(pipe.score(X_test,y_test))
    new_X = (np.array([700,777,5,1.0,500,5,100,20])).reshape(-1, 8)
    print(pipe.predict_proba(new_X))
    return pipe 



X,y = open_X_y(PATH)
X,y = sampling(X,y)
model(X,y)