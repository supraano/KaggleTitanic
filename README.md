# KaggleTitanic
Data Exploration, Preprocessing and Model Training for the Kaggle Challenge "Machine Learning from Disaster".

Two model classes were used to solve this problem:
- Support Vector Classifier
- Random Forest Classifier

Preprocessing:
- Encoding of feature 'sex'
- One-hot encoding for first letter of feature 'Cabin' and for feature 'Embarked'
- Missing value strategy for features 'Age' and 'Fare': Median
- Log transformation for feature 'Fare'
- Standard scaling for numerical features 'Fare' and 'Age'


For both models, hyperparameter optimization via Grid Search was done.
- SVC Parameters: C, kernel
- RFC Parameters: n_estimators, max_depth, max_features

Results (after submitting predictions in csv-format):
- SVC: 0.76555
- RFC: 0.75358

Looking forward to make improvements in the future. :)

Scripts:
- data_exploration.py -> visualizations and data exploration
- preprocessing.py -> all preprocessing steps that were done
- tune_train_test_rf.py -> Tuning, training and predictions for Random Forest Classifier
- tune_train_test_svm.py -> Tuning, training and predictions for Support Vector Classifier

Link to challenge: https://www.kaggle.com/competitions/titanic/overview
