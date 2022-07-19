import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv("../resources/train.csv")
test_df = pd.read_csv("../resources/test.csv")
labels = pd.read_csv("../resources/gender_submission.csv")

# combine data, to do easier preprocessing and data analysis
train_df["train"] = 1
test_df["train"] = 0
train_test = pd.concat([train_df, test_df])

# drop name and ticket column
train_test = train_test.drop(["Name"], axis=1)
train_test = train_test.drop(["Ticket"], axis=1)

print(train_test.head(5))

# encode sex
train_test["Sex"] = train_test["Sex"].apply(lambda x: 1 if x == "female" else 0)
print("Encoded feature 'sex':")
print(train_test["Sex"][:5])

# take first letter for cabin, one hot encoding
train_test["Cabin"] = train_test["Cabin"].apply(lambda x: "null" if pd.isna(x) else x[0])
train_test["Cabin"].unique()
train_test = pd.get_dummies(train_test, columns=["Cabin"])

# fill nan
train_test["Age"] = train_test["Age"].fillna(train_test["Age"].median())
train_test["Fare"] = train_test["Fare"].fillna(train_test["Fare"].median())

print(train_test.info())

# drop rows if 'Embarked' is missing (2 rows overall)
train_test.dropna(subset=["Embarked"], inplace=True)
train_test = pd.get_dummies(train_test, columns=["Embarked"])

# log transform
train_test["Fare"] = np.log(train_test["Fare"]+1) # +1 for -inf -> divide by zero

# divide dataset again
train = train_test[train_test["train"] == 1]
test = train_test[train_test["train"] == 0]
test = test.drop("train", axis=1)
train = train.drop("train", axis=1)

# passenger id not needed for train set
train = train.drop("PassengerId", axis=1)

# survived columns is integer
train['Survived'] = train['Survived'].astype(int)

print(train.columns)

# data scaling using standard scaler
scaler = StandardScaler()
scaled_features = ['Age', 'Fare']
train[scaled_features] = scaler.fit_transform(train[scaled_features])

# scale test data with train parameters
test[scaled_features] = scaler.transform(test[scaled_features])

# save as csv
test.to_csv('../resources/test_preprocessed.csv', index=False, sep=';')
train.to_csv('../resources/train_preprocessed.csv', index=False, sep=';')