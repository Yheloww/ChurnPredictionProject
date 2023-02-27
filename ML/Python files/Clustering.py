from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle

import pandas as pd

PATH = './Cleaned_ML_Unknonw.csv'

def open_df(PATH : str):
    """
    """
    data = pd.read_csv(PATH)
    data.drop(['Unnamed: 0'], axis=1, inplace=True)

    return data


def pipeline(data : pd.DataFrame, path_model: str):
    """
    """
    # separate categorical column and numerical for preprocess
    cat_cols = ['Gender', 'Education_Level','Card_Category', 'Income_Category','Attrition_Flag','Marital_Status']
    int_cols = data.drop(cat_cols, axis=1).columns
    # preprocessing step for the pipeline
    preprocess = ColumnTransformer(
        [
        ("Scaling", StandardScaler(), int_cols),
        ("OneHot", OneHotEncoder(sparse=True), cat_cols)
        ]
    )

    model = KMeans(n_clusters=3,n_init=10,init='k-means++')
    # pipeline 
    steps = [
        ("preprocess" , preprocess),
        ("PCA" , PCA(n_components=20)),
        ("model", model)
    ]
    pipe = Pipeline(steps)
    # fitting the model
    pipe.fit(data)
    pickle.dump(pipe, open( path_model, 'wb'))
    # Adding a cluster col to a new dataframe
    cluster_labels = pd.Series(model.labels_, name='cluster')
    data = data.join(cluster_labels.to_frame())
    data.to_csv('./clusters_3.csv')

# sepcifying path for the pickle model
clustering_model = '../Models/Churn_cluster_3.pkl'

pipeline(open_df(PATH), clustering_model)