import pandas as pd
import numpy as np 

# 201
# 다음은 Boston Housin 데이터 세트이다. 범죄율 컬럼인 Crim 항목의 상위에서 10번째 값
# (즉, 범죄율을 큰 순서로 내림차순 정렬했을 때 10번째에 위치한 값)으로 상위 10개의 값을 변환한 후,
# age가 80이상인 데이터를 추출하여 Crim의 평균값을 계산하시오. 
df = pd.read_csv("C:\\Users\\user\Desktop\\Python\\01\\01 시험예제\\201_boston.csv")

df = df.sort_values(by=['CRIM'], ascending = False).reset_index(drop=True)
print(df.head(10))

_10th = df.iloc[9,0]
# _10th = df.loc[9, 'CRIM']
print(_10th)

df.iloc[:10, 0] = _10th
# df.loc[:9, 'CRIM] = _10th
print(df.head(10))

target = df[df.AGE >= 80]
answer = target.CRIM.mean()
print(answer)

# 202
# 다음은 California Housing 데이터 세트이다. 주어진 데이터의 첫 번째 행부터 순서대로 80%까지의
# 데이터를 훈련 데이터로 추출 후, 전체 방 개수 컬럼을 의미하는 'total bedrooms'변수의 결측치를 
# 이 컬럼의 중앙값으로 대체한 데이터 세트를 구성한다. 결측치 대체 전의 'total bedrooms' 변수 표준편차
# 값과 대체 후의 표준편차 값의 차이에 대한 절대값을 계산하시오. 
df = pd.read_csv("C:\\Users\\user\Desktop\\Python\\01\\01 시험예제\\202_housing.csv")
print(df.head())

eight = df.shape[0] * 0.8
training = df.iloc[:int(eight)]

before = training['total_bedrooms'].std()

training.loc[training['total_bedrooms'].isnull()] = training['total_bedrooms'].median()
# training = training['total_bedrooms'].fillna(training['total_bedrooms'].median())
after = training['total_bedrooms'].std()
print(before, after)

answer = before - after
print(answer)

# 203
# 2번 문항에서 활용한 데이터세트를 그대로 활용한다. 인구 컬럼인 population 항목의 이상값의 합계를 계산하시오.
# (이상값은 평균에서 1.5*표준편차를 초과하거나 미만인 값의 범위로 정한다.)
df = pd.read_csv("C:\\Users\\user\Desktop\\Python\\01\\01 시험예제\\203_housing.csv")
ss = df.population.std()
mean = df.population.mean()

lower = mean - (1.5 * ss)
upper = mean + (1.5 * ss)
wrong = df[(df.population > upper ) | (df.population < lower)]
print(wrong)

answer = wrong.population.sum()
print(answer)

# 301
# 다음은 California Housing 데이터 세트이다. 데이터 중 결측치가 있는 경우 해당 데이터의 행을 모두
# 제거하고, 첫 번재 행부터 순서대로 70%의 데이터를 훈련 데이터로 추출한 데이터 세트를 구성한다.
# 변수 중 'housing_median_age'의 Q1 값을 정수로 계산하시오.
df = pd.read_csv("C:\\Users\\user\Desktop\\Python\\01\\01 시험예제\\301_housing.csv")

target = df.dropna()
train = target.iloc[:int(target.shape[0] * 0.7)]
answer = int(np.percentile(train['housing_median_age'], 25))
print(answer)

# 302
# 다음은 국가별 연도별 인구 10만 명당 결핵 유병률 데이터 세트이다. 2000년도의 국가별 결핵 유병률 데이터 세트에서
# 2000년도의 평균값보다 더 큰 유병률 값을 가진 국가의 수를 계산하시오.
df = pd.read_csv("C:\\Users\\user\Desktop\\Python\\01\\01 시험예제\\302_worlddata.csv")
target = df[df.year == 2000].drop(['year'], axis = 1)
target.index = ['value']
target = target.T
answer = len(target[target['value'] > target.value.mean()])
print(answer)

# 303
# 다음은 Titanic 데이터 세트이다. 주어진 데이터 세트의 컬럼 중 빈 값 또는 결측치를 확인하여, 결측치의 비율이
# 가장 높은 변수명을 출력하시오.
df = pd.read_csv("C:\\Users\\user\Desktop\\Python\\01\\01 시험예제\\303_titanic.csv")
# tt = pd.DataFrame(df.isnull().sum(), columns = ['Null value'])
# print(tt)
# tt['ratio'] = (tt['Null value'])/df.shape[0]
# print(tt[tt.ratio == tt.ratio.max()].index)  이렇게 하면 컬럼 값만 추출하기가 어려움

tt = df.isnull().sum().reset_index()
tt.columns = ['index','null']
tt['ratio'] = (tt.null)/df.shape[0]
answer = tt.loc[tt.ratio == tt.ratio.max(),'index'].values[0]
print(answer)

# 401
# 주어진 리스트에 대해 아래 과정을 차례로 수행한 최종 결괏값을 출력하시오
lst = [2, 3, 3.2, 5, 7.5, 10, 11.8, 12, 23, 25, 31.5, 34]
# 1. 제1사분위수와 제3사분위수를 구하시오
q1 = np.percentile(lst, 25)
q3 = np.percentile(lst, 75)
print(q1, q3)
# 2. 제1사분위수와 제 3사분위수 차이의 절댓값을 구하시오
absolute = abs(q1-q3)
print(absolute)
# 3. 그 값의 소수점을 버린 후 정수로 출력하시오
answer = int(absolute)
print(answer)

# 402
# 주어진 facebook 데이터 세트는 페이스북 라이브에 대한 사용자 반응을 집계한 것이다. 이 중
# love 반응(num_loves)과 wow 반응(num_wows)을 매우 긍정적인 반응이라고 정의할 때, 
# 전체 반응 수(num_reaction) 중 매우 긍정적인 반응 수가 차지하는 비율을 계산하시오.
# 그리고 그 비율이 0.5보다 작고 0.4보다 크며 유형이 비디오에 해당하는 건수를 정수로 출력하시오.
df = pd.read_csv("C:\\Users\\user\Desktop\\Python\\01\\01 시험예제\\402_facebook.csv")
print(df.head())
df['ratio'] = (df.num_loves + df.num_wows)/df.num_reactions
answer = len(df[(df.ratio > 0.4) & (df.ratio < 0.5) & (df.status_type == 'video')])
print(answer)

# 403
# 주어진 netflix 데이터 세트는 넷플릭스에 등록된 컨텐츠의 메타 데이터이다. 2018년 1월에 넷플릭스에
# 등록된 컨텐츠 중에서 'United kingdom'이 단독 제작한 컨텐츠의 수를 정수로 출력하시오. 
df = pd.read_csv("C:\\Users\\user\Desktop\\Python\\01\\01 시험예제\\403_netflix.csv")
print(df.info())
print(df.date_added)
target = df[df.date_added.str.contains('January') & df.date_added.str.contains('2018')]
answer = len(target[target.country == 'United Kingdom'])
print(answer)

# 501
# 주어진 trash bag 데이터 세트는 지역별 종량제 봉투 가격을 나타낸다. 가격 컬럼은 각 행의 조건을 
# 만족하는 해당 용량의 종량제 봉투가 존재하면 가격을 값으로, 존재하지 않으면 0을 값으로 갖는다.
# 이 때, 용도가 '음식물쓰레기'이고 '사용 대상이 '가정용'인 2L 봉투 가격의 평균을 소수점을 버린 후 
# 정수로 출력하여라.
df = pd.read_csv("C:\\Users\\user\\Desktop\\Python\\01\\01 시험예제\\501_trash_bag.csv", encoding='cp949')
target = df[(df.용도 == '음식물쓰레기') & (df.사용대상 == '가정용') & (df['2L가격'] != 0)]
answer = int(target['2L가격'].mean())
print(answer)

# 502
# BMI지수는 몸무게(kg)를 키(m)의 제곱으로 나누어 구하며, BMI 값에 따른 비만도 분류는 다음과 같다
# \\ 18.5 미만 '저체중' \\
# \\ 18.5 이상 23 미만 '정상' \\
# \\ 23 이상 25 미만 '과체중' \\
# \\ 25이상 30 미만 '경도비만' \\
# \\ 30 이상 '중등도비만' \\
# 이때 주어진 BMI데이터 세트에서 비만도가 정상에 속하는 인원 수와 과체중에 속하는 인원 수의 차이를
# 정수로 출력하시오. 
df = pd.read_csv("C:\\Users\\user\\Desktop\\Python\\01\\01 시험예제\\502_bmi.csv")
df['bmi'] = (df.Weight)/ ((df.Height / 100) ** 2)
nomal = len(df[(df.bmi < 23) & (df.bmi >= 18.5)])
over = len(df[(df.bmi >= 23) & (df.bmi < 25)])
answer = nomal - over
print(answer)

# 503 
# 주어진 students 데이터 세트는 각 학교의 학년별 총 전입학생, 총 전출학생, 전체 학생 수를 나타낸다.
# 순 전입학생 수는 총 전입학생 수에서 총 전출학생 수를 빼서 구할 수 있다. 
# 순 전입학생이 가장 많은 학교의 전체 학생 수를 구하시오. 
df = pd.read_csv("C:\\Users\\user\\Desktop\\Python\\01\\01 시험예제\\503_students.csv", encoding='cp949')
df['순 전입학생'] = df['총 전입학생'] - df['총 전출학생']
answer = df.groupby(['학교'])[['순 전입학생','전체 학생 수']].sum().values[0][1]
print(answer)
