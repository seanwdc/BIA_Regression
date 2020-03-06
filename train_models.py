import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

one_hot_df = pd.read_csv('Data/one_hot_df.csv')
label_df = pd.read_csv('Data/label_df.csv')

# RANDOM FOREST MODEL
y = label_df['SalePrice']
X_train,X_test,y_train,y_test = train_test_split (label_df,y,test_size =0.2)

regressor = RandomForestRegressor (n_estimators=300, random_state = 10)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
y_pred_train = regressor.predict (X_train)
#Find MSE 
MSE_test = metrics.mean_squared_error (y_test,y_pred)
MSE_train = metrics.mean_squared_error (y_train,y_pred_train)
print('The TRAIN RMSE for our Model is ', round(math.sqrt(MSE_train),10))
print('The TEST RMSE for our RF Model is ', round(math.sqrt(MSE_test),10))


# SEAN
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

