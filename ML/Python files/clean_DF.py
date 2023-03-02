import pandas as pd
import numpy as np

# open csv
churn = pd.read_csv('./BankChurners.csv')
# churned = pd.read_csv('../Data/clusters.csv')

# replace and get rid of nan


def clean_nan(data: pd.DataFrame):
    """
    """
    print(data.shape)
    data = data.replace(['Unknown'], 0)
    # data.dropna(inplace=True)
    print(data.shape)

    print('clean ok')
    return data


def col(data):
    """
    """
    # get rid of useless coluumns
    data.rename(columns={
                'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1': 'class 1'}, inplace=True)
    data.rename(columns={
                'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2': 'class 2'}, inplace=True)
    data.drop(['class 1', 'class 2', 'CLIENTNUM'], axis=1, inplace=True)
    print('col ok ')
    return data


def save(data, name):
    """
    """
    data.to_csv(name)
    print("saved")
    return data


# (save(col(clean_nan(churned)), 'Cleaned_clusters_Unknown.csv')).head()
save(col(clean_nan(churn)), 'cleaned_df.csv')
