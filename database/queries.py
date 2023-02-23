from pymongo import MongoClient
import pandas as pd
from os import path

client = MongoClient(host="localhost", port=27017)

db = client["bankChurners"]

print(db.list_collection_names())

df = pd.DataFrame

filename = path.join(path.abspath("Database"),"BankChurners.csv")
with (open(filename, "r")) as file :
    df = pd.read_csv(file)

print(df.Attrition_Flag.unique())

#Used ones to clean the useless columns
'''df = df.drop('Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', axis=1)
df = df.drop('Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2', axis=1)

with (open(filename, "w")) as file :
    file.write(df.to_csv())'''
