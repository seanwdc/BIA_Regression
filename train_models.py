import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import kfold
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

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