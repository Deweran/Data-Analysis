import pandas as pd 
import numpy as np 

x_test = pd.read_csv('C:\\Users\\김이슬아\\Desktop\\python\\빅분기 실기 자료\\504_x_test.csv')
x_train = pd.read_csv('C:\\Users\\김이슬아\\Desktop\python\\빅분기 실기 자료\\504_x_train.csv')
y_train = pd.read_csv('C:\\Users\\김이슬아\\Desktop\python\\빅분기 실기 자료\\504_y_train.csv')

print(x_train.info())
print(x_train.head(3))

col_del = ['ID']
num = ['year','mileage','tax','mpg','engineSize']
obj = ['model','transmission','fuelType']
y = ['price']

from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit(x_train[num])

x_train[num] = scaler.transform(x_train[num])
x_test[num] = scaler.transform(x_test[num])

from sklearn.preprocessing import LabelEncoder 
x = pd.concat([x_train[obj], x_test[obj]])
le = LabelEncoder()

for col in obj:
    le = LabelEncoder()
    le.fit(x[col])
    x_train[col] = le.transform(x_train[col])
    x_test[col] = le.transform(x_test[col])

    print(col)
    print(le.classes_)
    print('\n')

from sklearn.model_selection import train_test_split
x_tr, x_val, y_tr, y_val = train_test_split(x_train[num + obj], y_train[y].values.ravel(), test_size = 0.3)

# model 
from sklearn.ensemble import RandomForestRegressor 
model = RandomForestRegressor(max_depth = 8, n_estimators = 300, random_state = 120)
model.fit(x_tr, y_tr)

# hyper parameter tuning 
#from sklearn.model_selection import GridSearchCV
#parameters={'n_estimators':[50,100,300],'max_depth':[4,6,8]}
#model3 = RandomForestRegressor()
#clf = GridSearchCV(estimator = model3, param_grid = parameters, cv = 3)
#clf.fit(x_tr,y_tr)
#print('최적의 파라미터: ',clf.best_params_)

# assessment 
from sklearn.metrics import mean_squared_error
model_a = model.predict(x_val)

def calculate(actual, pred):
    return np.sqrt(mean_squared_error(actual, pred))

score = calculate(y_val, model_a)
print('randomforest = ', score)

pred = model.predict(x_test[num + obj])
result = pd.DataFrame({'ID':x_test.id, 'price': pred})
print(result)
# result.to_csv('000003030.csv', index = False)