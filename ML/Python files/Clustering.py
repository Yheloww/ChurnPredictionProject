from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import pandas as pd

PATH = './Cleaned_ML_Unknonw.csv'

def open(PATH : str):
    """
    """
    data = pd.read_csv(PATH)
    return data


def pipeline(data : pd.DataFrame):
    """
    """
    cat_cols = ['Gender', 'Education_Level','Card_Category', 'Income_Category']
    int_cols = data.drop(cat_cols, axis=1).columns
    print(int_cols)

    preprocess = ColumnTransformer(
        [
        ("Scaling", StandardScaler(), int_cols),
        ("OneHot", OneHotEncoder(sparse=True), cat_cols)
        ]
    )

    model = KMeans(n_clusters=3,n_init=10,init='k-means++')

    steps = [
        ("preprocess" , preprocess),
        ("model", model),
        ("PCA" , PCA(n_components=20))
    ]
    pipe = Pipeline(steps)

    cluster_labels = pd.Series(pipe.labels_, name='cluster')
    data = data.join(cluster_labels.to_frame())
    
    data.to_csv('clusters.csv')
    print(data.head())

    pipe

pipeline(open(PATH))