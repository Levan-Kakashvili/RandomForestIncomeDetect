def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

#Below we create example person to predict income based on features
person = [26, 0, 0, 40, 0, 1, 0]
person_data = pd.DataFrame([person], columns=["age", "capital-gain", "capital-loss", "hours-per-week", "sex-int", "country-int", "race-int"])

income_data = pd.read_csv("income.csv", header = 0, delimiter = ", ", na_values=['?'])
income_data.fillna(0, inplace=True)

#We need to change Sex values from Str to Int by changing Male to 0 and Female to 1, same for countries like US and others 0-1
income_data["sex-int"] = income_data["sex"].apply(lambda row: 0 if row == "Male" else 1)
income_data["country-int"] = income_data["native-country"].apply(lambda row: 0 if row == "United-States" else 1)
income_data["race-int"] = income_data["race"].apply(lambda row: 0 if row == "White" else 1)
#print(income_data.head(0))
#print(income_data["race"].value_counts())
labels = income_data[["income"]]
data = income_data[["age", "capital-gain", "capital-loss", "hours-per-week", "sex-int", "country-int", "race-int"]]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)

forest = RandomForestClassifier(random_state = 1)
forest.fit(train_data, train_labels)
#to print wich feature has more impact on a results
print(forest.feature_importances_)
print(forest.score(test_data, test_labels))
print(forest.predict(person_data))

