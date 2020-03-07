import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

one_hot_df = pd.read_csv('Data/one_hot_df.csv')
label_df = pd.read_csv('Data/label_df.csv')

# Containing all variables
y = one_hot_df['SalePrice']
X = one_hot_df.drop(['SalePrice'], axis =1)
# print(X_train.shape)

model = LinearRegression()
metrics = cross_validate(model, X,y,cv = 10, scoring = ('neg_root_mean_squared_error'))
lr_cv = -metrics['test_score'].mean()
print(lr_cv)

parameters = {'alpha': [0.1, 0.5, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]} 
rr = Ridge()
metrics = cross_validate(rr, X, y, cv = 10, scoring = ('neg_root_mean_squared_error'))
print(-metrics['test_score'].mean())

test = GridSearchCV(Ridge(), parameters, scoring = 'neg_root_mean_squared_error', cv = 5)
test.fit(X,y)
# print(test.best_params_)
print(-test.best_score_)