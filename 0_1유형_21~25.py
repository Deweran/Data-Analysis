import pandas as pd
import numpy as np
import datetime

# 사기회사 분류 데이터
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/audit/train.csv")
print(df.head(3))

# 1. 데이터의 Risk 값에 따른 score_a와 score_b의 평균값을 구하여라
ans = df.groupby(['Risk'])[['Score_A','Score_B']].mean()
print(ans)


# 센서데이터 동작유형 분류 데이터
sensor = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/muscle/train.csv")
print(sensor.head(3))

# 1. pose값에 따른 각 motion컬럼의 중간값의 가장 큰 차이를 보이는 motion컬럼은 어디이며 그값은?
t = sensor.groupby('pose').median().T
dfs = abs(t[0] - t[1]).sort_values().reset_index()
answer = dfs[dfs[0] == dfs[0].max()]['index'].values
print(t)
print(dfs)
print(answer)

# 현대 차량 가격 분류문제 데이터
price = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/hyundai/train.csv")
print(price.head(3))

# 1. 정보(row수)가 가장 많은 상위 3차종의 price값의 각 평균값은?
lst = list(price.groupby(['model']).size().sort_values(ascending = False).index[:3])
answer = price[price.model.isin(lst)].groupby(['model'])['price'].mean()
print(answer)


# 당뇨여부판단 데이터
diabetes = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/diabetes/train.csv")
print(diabetes.head(3))

# 1. Outcome 값에 따른 각 그룹의 각 컬럼의 평균 차이를 구하여라
tt = diabetes.groupby(['Outcome']).mean().T
tt['mean'] = abs(tt[0] - tt[1])
answer = tt['mean']
print(answer)


# 넷플릭스 주식 데이터
netflex = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/nflx/NFLX.csv")
print(netflex.head(3))

# 1. 매년 5월달의 open가격의 평균값을 데이터 프레임으로 표현하라
netflex['Date']  = pd.to_datetime(netflex['Date'])
target = netflex.groupby(netflex['Date'].dt.strftime('%Y-%m')).mean()
answer = target.loc[target.index.str.contains('-05')].Open
print(answer)

