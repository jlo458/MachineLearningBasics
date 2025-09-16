# General techniques for cleaning data using pandas 

import pandas as pd 

# Index_col determines which column used as index
raw_data = pd.read_csv("./titanic.csv", index_col="PassengerId")

# tablename.to_csv("filename.csv", index=none)  # To save new dataset
# tablename = pd.read_csv("filename", index_col=whatever)  # For loading


# Shows how many NaN's there are (Not a Number), if too many, scrap the column
#print(raw_data.isna().sum())

# Scraps/"drops" line of data (col if axis=1, row if axis=0)
clean_data = raw_data.drop("Cabin", axis=1)
#print(clean_data.isna().sum()) 


# Can fill data instead (usually with mean/median) 
med_age = clean_data["Age"].mean()
clean_data["Age"] = clean_data["Age"].fillna(med_age) 

# Can also fill with filler class, if only a few NaN's (e.g. U for unknown), where value is qualitative
clean_data["Embarked"] = clean_data["Embarked"].fillna('U')
print(clean_data["Embarked"])  

clean_data.to_csv("./titanic_clean.csv", index=None)

# Makng the data useful (turning categorical into quantative etc) 


import pandas as pd 

data = pd.read_csv("./preprocessed_data2.csv", index_col=None) 

'''classCols = pd.get_dummies(data['Pclass'], prefix='Pclass')
data = pd.concat([data, classCols], axis=1)
data = data.drop(['Pclass'], axis=1)

data.to_csv("./preprocessed_data2.csv", index=None)'''


'''genderCols = pd.get_dummies(data['Sex'], prefix='sex') 
embarkedCols = pd.get_dummies(data["Embarked"], prefix='embarked')

data = pd.concat([data, genderCols], axis=1)
data = pd.concat([data, embarkedCols], axis=1)

data = data.drop(['Sex', 'Embarked'], axis=1)

print(data)

data.to_csv("./preprocessed_data.csv", index=None)'''
