import pandas as pd
import numpy as np
import datetime 

# 지역구 에너지 소비량 데이터
df= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/consum/Tetuan%20City%20power%20consumption.csv')
print(df.head(3))

# 1. DateTime컬럼을 통해 각 월별로 몇개의 데이터가 있는지 데이터 프레임으로 구하여라
df['DateTime'] = pd.to_datetime(df.DateTime)
answer = df.DateTime.dt.month.value_counts().sort_index().to_frame()
print(answer)

# 2. 3월달의 각 시간대별 온도의 평균들 중 가장 낮은 시간대의 온도를 출력하라
target = df[df.DateTime.dt.month == 3]
answer = target.groupby(target.DateTime.dt.hour)['Temperature'].mean().min()
print(answer)

# 3. 3월달의 각 시간대별 온도의 평균들 중 가장 높은 시간대의 온도를 출력하라
answer = target.groupby(target.DateTime.dt.hour)['Temperature'].mean().max()
print(answer)

# 4. Zone 1 Power Consumption 컬럼의 value값의 크기가 Zone 2 Power Consumption 컬럼의 
# value값의 크기보다 큰 데이터들의 Humidity의 평균을 구하여라
target = df[df['Zone 1 Power Consumption'] > df['Zone 2  Power Consumption']]
answer = target.Humidity.mean()
print(answer)

# 5. 각 zone의 에너지 소비량의 상관관계를 구해서 데이터 프레임으로 표기하라
result = df.iloc[:,-3:].corr()
print(result)

# 6. Temperature의 값이 10미만의 경우 A, 10이상 20미만의 경우 B,20이상 30미만의 경우 C, 
# 그 외의 경우 D라고 할때 각 단계의 데이터 숫자를 구하여라
def stratify(x):
    if x < 10:
        return 'A'
    elif 10 <= x < 20:
        return 'B'
    elif 20 <= x < 30:
        return 'C'
    else:
        return 'D'

df['stratify'] = df['Temperature'].map(stratify)
answer = df['stratify'].value_counts()
print(answer)

# 7. 6월 데이터중 12시의 Temperature의 표준편차를 구하여라
target = df[(df.DateTime.dt.month == 6) & (df.DateTime.dt.hour == 12)]
answer = target.Temperature.std()
print(answer)

# 8. 6월 데이터중 12시의 Temperature의 분산을 구하여라 
answer = target.Temperature.var()
print(answer)

# 9. Temperature의 평균이상의 Temperature의 값을 가지는 데이터를 Temperature를 기준으로 정렬 했을때 
# 4번째 행의 Humidity 값은?
temp_mean = df.Temperature.mean()
target = df[df.Temperature >= temp_mean].sort_values('Temperature')
answer = target.Humidity.values[3]
print(answer)

# 10. Temperature의 중간값 이상의 Temperature의 값을 가지는 데이터를
# Temperature를 기준으로 정렬 했을때 4번째 행의 Humidity 값은?
temp_median = df.Temperature.median()
target = df[df.Temperature >= temp_median].sort_values('Temperature')
answer = target.Humidity.values[3]
print(answer)


# 포켓몬 정보 데이터
df2 = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/pok/Pokemon.csv')
print(df2.head(3))

# 1. Legendary 컬럼은 전설포켓몬 유무를 나타낸다. 전설포켓몬과 그렇지 않은 포켓몬들의 HP평균의 차이를 구하여라
target_l = df2[df2.Legendary == False]['HP'].mean()
target_n = df2[df2.Legendary == True]['HP'].mean()
answer = (target_n - target_l)
print(answer)

# 2. Type 1은 주속성 Type 2 는 부속성을 나타낸다. 가장 많은 부속성 종류는 무엇인가?
target = df2['Type 2'].value_counts().index[0]
print(target)

# 3. 가장 많은 Type 1 의 종의 평균 Attack 을 평균 Defense로 나눈값은?
target = df2[df2['Type 1'] == df2['Type 1'].value_counts().index[0]]
answer = (target.Attack.mean()) / (target.Defense.mean())
print(answer)

# 4. 포켓몬 세대(Generation) 중 가장많은 Legendary를 보유한 세대는 몇세대인가?
first = df2[df2.Legendary == True]
target = first.groupby(['Generation']).size().sort_values(ascending=False).index[0]
print(target)

# 5. ‘HP’, ‘Attack’, ‘Defense’, ‘Sp. Atk’, ‘Sp. Def’, ‘Speed’ 간의 상관 계수 중 
# 가장 절댓값이 큰 두 변수와 그 값을 구하여라
target = df2[[ 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].corr()\
    .unstack().reset_index().rename(columns={0: "corr"})
result = target[target['corr'] != 1].sort_values('corr',ascending=False).iloc[0]
print(result)

# 6. 각 Generation과 Attack으로 오름차순 정렬 시 상위 3개 데이터들(18개)의 Attack의 전체 평균을 구하여라
answer = df2.sort_values(['Generation','Attack']).groupby(['Generation'])['Attack'].head(3).mean()
print(answer)

# 7. 각 Generation과 Attack으로 내림차순 정렬시 상위 5개 데이터들(30개)의 Attack의 전체 평균을 구하여라
answer = df2.sort_values(['Generation','Attack'], ascending=False)\
    .groupby(['Generation'])['Attack'].head(5).mean()
print(answer)

# 8. 가장 흔하게 발견되는 (Type1 , Type2) 의 쌍은 무엇인가?
result = df2.groupby(['Type 1','Type 2']).size().sort_values(ascending=False).index[0]
print(result)

# or 
#result = df2[['Type 1','Type 2']].value_counts().head(1)
#print(result)

# 9. 한 번씩만 존재하는 (Type1 , Type2)의 쌍의 갯수는 몇개인가?
target = df2[['Type 1','Type 2']].value_counts()
result = len(target[target == 1])
print(result)

# 10. 한 번씩만 존재하는 (Type1 , Type2)의 쌍을 각 세대(Generation)은 각각 몇 개씩 가지고 있는가?
target = df2[['Type 1','Type 2']].value_counts()
target2 = target[target == 1]

lst = []
for value in target2.reset_index().values:
    t1 = value[0]
    t2 = value[1]
    
    sp = df2[(df2['Type 1'] == t1) & (df2['Type 2'] == t2)]
    lst.append(sp)

result = pd.concat(lst).reset_index(drop = True).Generation.value_counts().sort_index()
print(result)
