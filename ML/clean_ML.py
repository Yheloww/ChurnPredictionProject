import pandas as pd
import numpy as np

# open csv
churn = pd.read_csv('./BankChurners.csv')

# replace and get rid of nan 
churn = churn.replace(['Unknown'], np.nan)
churn.dropna(inplace=True)

# get rid of useless coluumns
churn.rename(columns = {'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1':'class 1'}, inplace = True)
churn.rename(columns = {'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2':'class 2'}, inplace = True)
churn.drop(['class 1', 'class 2','CLIENTNUM'], axis=1, inplace=True)

# replacing categorical values for logical int
churn = churn.replace(['Existing Customer','Married'], 1)
churn = churn.replace(['Attrited Customer','Single'], 0)
churn = churn.replace(['Divorced'], 2)
churn = churn.replace(['Uneducated','High School','College','Graduate','Post-Graduate','Doctorate'], [0,1,2,3,4,5])
churn = churn.replace(['Less than $40K','$40K - $60K','$60K - $80K','$80K - $120K','$120K +'], [0,1,2,3,4])
churn = churn.replace(['Blue','Silver','Gold','Platinum'], [0,1,2,3])
churn = pd.get_dummies(churn, columns=['Gender'])

# sav the csv 
churn.to_csv('cleaned_ML.csv')