from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle

import numpy as np 
import pandas as pd

PATH = './cleaned_df.csv'

def open_df(PATH : str):
    """
    """
    data = pd.read_csv(PATH)
    # choosing only important features
    features = data[['Total_Trans_Ct', 'Total_Revolving_Bal', 'Total_Relationship_Count',
                   'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Ct_Chng_Q4_Q1', 'Avg_Open_To_Buy', 'Customer_Age']]

    return features


def pipeline(data : pd.DataFrame):
    """
    """
    # separate categorical column and numerical for preprocess
    int_cols = data.columns
    # preprocessing step for the pipeline
    preprocess = ColumnTransformer(
        [
        ("Scaling", StandardScaler(), int_cols),
        ]
    )
    model = KMeans(n_clusters=3,n_init=10,init='k-means++',random_state=1)
    # pipeline 
    steps = [
        ("preprocess" , preprocess),
        ("PCA" , PCA(n_components=2, random_state=1)),
        ("model", model)
    ]
    pipe = Pipeline(steps)
    # fitting the model
    pipe.fit(data)
    # save new df
    data_base = pd.read_csv(PATH)
    cluster_labels = pd.Series(model.labels_, name='cluster')
    data_base = data_base.join(cluster_labels.to_frame())
    data_base.to_csv('./clusters_3_final.csv')

    return pipe, data_base

def prediction(pipe,  path_model: str, data : pd.DataFrame):
    """
    """
    # prediction
    new_X = data.iloc[[55]]
    prediction = pipe.predict(new_X)[0]
    dict_percentages = {0: 8.1, 1 : 5.3, 2 : 30.1}
    perc = dict_percentages[prediction]
    print(f'the client is from cluster {prediction} then his probability of churning is {perc} %')
    # savec pickle
    pickle.dump(pipe, open( path_model, 'wb'))
    return 


# sepcifying path for the pickle model
clustering_model = '../Models/Churn_cluster_3_final.pkl'
# steps to save df and predict cluster
data = open_df(PATH)
pipe, data_base = pipeline(data)
prediction(pipe, clustering_model, data_base)
