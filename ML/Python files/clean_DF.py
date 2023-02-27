import pandas as pd
import numpy as np

# open csv
churn = pd.read_csv('./BankChurners.csv')
#churned = pd.read_csv('../Data/clusters.csv')

# replace and get rid of nan 
def clean_nan(data : pd.DataFrame): 

    print(data.shape)
    data = data.replace(['Unknown'], 0)
    #data.dropna(inplace=True)
    print(data.shape)

    print('clean ok')
    return data


def col(data):
    # get rid of useless coluumns
    data.rename(columns = {'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1':'class 1'}, inplace = True)
    data.rename(columns = {'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2':'class 2'}, inplace = True)
    data.drop(['class 1', 'class 2','CLIENTNUM'], axis=1, inplace=True)
    print('col ok ')
    return data

def save(data, name):
    data.to_csv(name)
    print("saved")
    return data 


def replace(data):
# replacing categorical values for logical int
    data = data.replace(['Existing Customer','Married'], 2)
    data = data.replace(['Attrited Customer','Single'], 1)
    data = data.replace(['Divorced'], 3)
    data = data.replace(['Uneducated','High School','College','Graduate','Post-Graduate','Doctorate'], [1,2,3,4,5,6])
    data = data.replace(['Less than $40K','$40K - $60K','$60K - $80K','$80K - $120K','$120K +'], [1,2,3,4,5])
    data = data.replace(['Blue','Silver','Gold','Platinum'], [1,2,3,4])
    data= pd.get_dummies(data, columns=['Gender'])
    return data 

#(save(col(clean_nan(churned)), 'Cleaned_clusters_Unknown.csv')).head()
print((save(col(clean_nan(replace(churn))), 'Cleaned_ML_Unknonw.csv')).head())