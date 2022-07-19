import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv("../resources/train.csv")
test_df = pd.read_csv("../resources/test.csv")
labels = pd.read_csv("../resources/gender_submission.csv")

print(train_df.head(5))
print(labels.head(5))

# combine data, to do easier preprocessing and data analysis
train_df["train"] = 1
test_df["train"] = 0
train_test = pd.concat([train_df, test_df])
print(train_test.head(5))

print(train_test.describe())
print(train_test.info())

# categorical and numeric columns
num = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
cat = ["Cabin", "Sex", "Ticket", "Embarked"]

# show histogram for numerical features
for col in num:
    plt.hist(train_test[col])
    plt.title(col)
    plt.show()

print(train_test[num].corr())

# show barplot for categorical features
for col in cat:
    plt.figure(figsize=(12, 12))
    sns.barplot(x=train_df[col].value_counts().index, y=train_df[col].value_counts()).set_title(col)
    plt.show()

