import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import pickle


PATH = "./clusters_3.csv"


def open_X_y(path: str, class_name : str):
    """
    """
    # opening the dataset
    churn = pd.read_csv(path)
    churn.drop(['Unnamed: 0'], axis=1, inplace=True)
    # selecting only important features
    if class_name == 'Attrition_Flag':
        churn = churn[['Attrition_Flag','Total_Trans_Ct', 'Total_Revolving_Bal', 'Total_Relationship_Count',
                    'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Ct_Chng_Q4_Q1', 'Avg_Open_To_Buy', 'Customer_Age']]
    else : 
        churn = churn[['cluster','Total_Trans_Ct', 'Total_Revolving_Bal', 'Total_Relationship_Count',
                        'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Ct_Chng_Q4_Q1', 'Avg_Open_To_Buy', 'Customer_Age']]
        churn = churn.dropna()

    less = churn.reindex(sorted(churn.columns), axis=1)

    #divide to feature and results
    y = less[class_name].values
    X = less.drop(class_name,axis=1).values
 
    return X,y

def split(X: np.array, y: np.array):
    """
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    return X_train, X_test, y_train, y_test


def sampling(X: np.array, y: np.array):
    """
    """
    ros = RandomOverSampler(sampling_strategy="not majority", random_state=42)
    X_ros, y_ros = ros.fit_resample(X,y)

    return X_ros,y_ros

def model(X : np.array,y : np.array,path_model : str):
    """
    """
    X_train, X_test, y_train, y_test = split(X,y)
    # oversampling
    X_train, y_train = sampling(X_train,y_train)
    # setting up the steps for the pipeline
    steps=[("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(random_state=45, max_depth=6,criterion='gini', max_features=None))]
    # training the model
    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)
    # save as pickle
    pickle.dump(pipe, open( path_model, 'wb'))
    # evaluation and prediction 
    print(pipe.score(X_test,y_test))
    # test of prediction
    new_X = (np.array([20, 100, 30, 500, 50, 2.500, 20, 40])).reshape(-1, 8)
    proba = pipe.predict_proba(new_X).tolist()
    prediction = pipe.predict(new_X)
    #print(classification_report(y_test, prediction))
    pred = prediction[0]
    if pred == 0: 
        pourcent = 6.0
    elif pred == 1:
        pourcent = 8.7
    else :
        pourcent = 33.8
    proba = proba[0][0]
    print(f"The customer is from the cluster {pred}, and then has {pourcent}% chance of churning")

    return pipe

def percentage(df):
    """
    """


chrun_model = '../Models/Churn_class.pkl'
chrun_model = '../Models/test_cluster_class.pkl'
X,y = open_X_y(PATH, 'cluster')
model(X,y, chrun_model)
