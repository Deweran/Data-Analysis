import pandas as pd
import numpy as np 

x_test = pd.read_csv('C:\\Users\\김이슬아\\Desktop\\python\\빅분기 실기 자료\\204_x_test.csv')
x_train = pd.read_csv('C:\\Users\\김이슬아\\Desktop\python\\빅분기 실기 자료\\204_x_train.csv')
y_train = pd.read_csv('C:\\Users\\김이슬아\\Desktop\python\\빅분기 실기 자료\\204_y_train.csv')

print(x_train.info())
print(x_train.head(3))

col_del = ['ID']
num = ['Customer_care_calls','Customer_rating','Cost_of_the_Product','Prior_purchases','Discount_offered',
       'Weight_in_gms']
obj = ['Warehouse_block','Mode_of_Shipment','Product_importance','Gender']
y = ['Reached.on.Time_Y.N']

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
x_tr, x_val, y_tr, y_val = train_test_split(x_train[num + obj], y_train[y].values.ravel(),
                                            test_size = 0.3, stratify = y_train[y].values.ravel())

# model 
from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier(n_estimators = 50, max_depth = 3, random_state = 123)
model.fit(x_tr, y_tr)

# assessment 
from sklearn.metrics import roc_auc_score
model_a = model.predict_proba(x_val)[:, 1]
score = roc_auc_score(y_val, model_a)
print(score) # 0.752

# submit
pred = model.predict_proba(x_test[num + obj])[:, 1]
result = pd.DataFrame({'ID': x_test.ID, 'pred': pred})
print(result)

# result.to_csv('00020202.csv', index = False)