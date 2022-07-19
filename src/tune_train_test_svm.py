import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

train_df = pd.read_csv('../resources/train_preprocessed.csv', sep=';')
test_df = pd.read_csv('../resources/test_preprocessed.csv', sep=';')

y_train, X_train = train_df.iloc[:, 0].to_numpy(), train_df.iloc[:, 1:].to_numpy()

parameters = {'kernel':('linear', 'rbf'), 'C': [i for i in range(1, 10, 1)]}

svc = SVC()
svc_models = GridSearchCV(svc, parameters, cv=3, scoring='balanced_accuracy')
svc_models.fit(X_train, y_train)

print(pd.DataFrame(svc_models.cv_results_)
      .sort_values(by='rank_test_score', ascending=True)[['params', 'mean_test_score', 'rank_test_score']])

# train on best parameter setting
best_params = svc_models.best_params_
svc_final = SVC(C=best_params['C'], kernel=best_params['kernel'])
svc_final.fit(X_train, y_train)

print(test_df)
predictions = svc_final.predict(test_df.iloc[:, 2:].to_numpy())
test_df['Survived'] = predictions
print(test_df)
test_df.iloc[:, :2].to_csv('../resources/predictions_svc.csv', sep=',', index=False)