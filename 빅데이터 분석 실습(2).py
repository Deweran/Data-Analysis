import seaborn as sns
import pandas as pd
import numpy as np
import warnings

#print(sns.get_dataset_names())
#data = sns.load_dataset('penguins')
#print(data.head(3))

warnings.filterwarnings('ignore')

x_test = pd.read_csv('C:/Users/USER/Desktop/Python work space/Data Analysis/penguin_X_test.csv')
x_train = pd.read_csv('C:/Users/USER/Desktop/Python work space/Data Analysis/penguin_X_train.csv')
y_train = pd.read_csv('C:/Users/USER/Desktop/Python work space/Data Analysis/penguin_y_train.csv')

#데이터 전처리
print(x_train.info())

train = pd.concat([x_train, y_train], axis = 1)
print(train.loc[(train.sex.isna()) | (train.bill_length_mm.isna()) | (train.bill_depth_mm.isna()) | (train.flipper_length_mm.isna()) | (train.body_mass_g.isna())])

train = train.dropna()
train.reset_index(drop=True, inplace=True)

x_train = train[['species','island', 'sex','bill_length_mm', 'bill_depth_mm','flipper_length_mm']]
y_train = train[['body_mass_g']]

print(x_train.describe())

col_del = []
col_num = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']
col_cat = ['species', 'island','sex']
col_y = ['body_mass_g']

x = pd.concat([x_train, x_test])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown='ignore')
ohe.fit(x[col_cat])

x_train_res = ohe.transform(x_train[col_cat])
x_test_res = ohe.transform(x_test[col_cat])

print(x_train_res)

x_train_ohe = pd.DataFrame(x_train_res.todense(), columns = ohe.get_feature_names_out())
x_test_ohe = pd.DataFrame(x_test_res.todense(), columns = ohe.get_feature_names_out())
print(x_train_ohe)

x_train_fin = pd.concat([x_train[col_num], x_train_ohe], axis=1)
x_test_fin = pd.concat([x_test[col_num], x_test_ohe], axis=1)

#데이터 모형 구축
from sklearn.model_selection import train_test_split
x_tr, x_val, y_tr, y_val = train_test_split(x_train_fin, y_train, test_size=0.3)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_tr[col_num])

x_tr[col_num] = scaler.transform(x_tr[col_num])
x_val[col_num] = scaler.transform(x_val[col_num])
x_test_fin[col_num] = scaler.transform(x_test_fin[col_num])

from sklearn.linear_model import LinearRegression

modelLR = LinearRegression()
modelLR.fit(x_tr, y_tr)

y_val_pred = modelLR.predict(x_val)
print(y_val_pred)

print(modelLR.intercept_)

coef = pd.Series(data = modelLR.coef_[0], index = x_train_fin.columns)
print(coef.sort_values())

#데이터 모형 평가
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_val, y_val_pred)
rmse = mean_squared_error(y_val, y_val_pred, squared=False)

print('MSE : {0:.3f}, RMSE : {1:.3F}'.format(mse, rmse))

y_pred = modelLR.predict(x_test_fin)
print(y_pred)

#pd.DataFrame({'body_mass_g': y_pred[:, 0]}).to_csv('C:/Users/USER/Desktop/Python work space/Data Analysis', index=False)


