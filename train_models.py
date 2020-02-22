import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import kfold
from sklearn import metrics

one_hot_df = pd.read_csv('Data/one_hot_df.csv')
label_df = pd.read_csv('Data/label_df.csv')
