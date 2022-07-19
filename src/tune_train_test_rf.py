import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV

train_df = pd.read_csv('../resources/train_preprocessed.csv', sep=';')
test_df = pd.read_csv('../resources/test_preprocessed.csv', sep=';')

y_train, X_train = train_df.iloc[:, 0].to_numpy(), train_df.iloc[:, 1:].to_numpy()

parameters = {'n_estimators': [i for i in range(100, 1000, 100)],
              'max_depth': [i for i in range(50, 300, 50)],
              'max_features': ['sqrt', 'log2']}

rfc = RFC()
rfc_models = GridSearchCV(rfc, parameters, cv=3, scoring='balanced_accuracy')
rfc_models.fit(X_train, y_train)

print(pd.DataFrame(rfc_models.cv_results_)
      .sort_values(by='rank_test_score', ascending=True)[['params', 'mean_test_score', 'rank_test_score']])

# train on best parameter setting
best_params = rfc_models.best_params_
rfc_final = RFC(n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                max_features=best_params['max_features'])
rfc_final.fit(X_train, y_train)

print(test_df)
predictions = rfc_final.predict(test_df.iloc[:, 2:].to_numpy())
test_df['Survived'] = predictions
print(test_df)
test_df.iloc[:, :2].to_csv('../resources/predictions_rfc.csv', sep=',', index=False)