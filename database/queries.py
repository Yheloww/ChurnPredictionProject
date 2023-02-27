from pymongo import MongoClient
import pandas as pd
from os import path
from sqlalchemy import create_engine, MetaData

df = pd.DataFrame

filename = path.abspath("bankchurners.csv")
with (open(filename, "r")) as file :
    df = pd.read_csv(file)

engine = create_engine('sqlite:///database/churn_prediction.db')
df.to_sql('clients', con=engine, if_exists='replace')



#Used ones to clean the useless columns
'''df = df.drop('Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', axis=1)
df = df.drop('Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2', axis=1)

with (open(filename, "w")) as file :
    file.write(df.to_csv())'''
