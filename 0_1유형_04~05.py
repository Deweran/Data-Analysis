import pandas as pd
import numpy as np 
import datetime 

# 서울시 따릉이 이용정보 데이터
data =pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bicycle/seoul_bi.csv')
print(data.head(3))

# 1. 대여일자별 데이터의 수를 데이터프레임으로 출력하고, 가장 많은 데이터가 있는 날짜를 출력하라
pro = data.groupby(['대여일자']).size().sort_values(ascending = False)
print(pro.index[0])

# or
#result = data['대여일자'].value_counts().sort_index().to_frame()
#answer = result[result.대여일자 == result.대여일자.max()].index[0]
#print(result)
#print(answer)

# 2. 각 일자의 요일을 표기하고 (‘Monday’ ~’Sunday’) ‘day_name’컬럼을 추가하고 
# 이를 이용하여 각 요일별 이용 횟수의 총합을 데이터 프레임으로 출력하라
data['대여일자'] = pd.to_datetime(data['대여일자'])
data['day_name'] = data['대여일자'].dt.day_name()

result = data.groupby(['day_name']).size()
print(result)

# or
#result =  data.day_name.value_counts().to_frame()
#print(result)

# 3. 각 요일별 가장 많이 이용한 대여소의 이용횟수와 대여소 번호를 데이터 프레임으로 출력하라
result = data.groupby(['day_name','대여소번호']).size().to_frame('size')\
    .sort_values(['day_name','size'],ascending=False).reset_index()
answer  = result.drop_duplicates('day_name',keep='first').reset_index(drop=True)
print(answer)

# 4. 나이대별 대여구분 코드의 (일일권/전체횟수) 비율을 구한 후 가장 높은 비율을 가지는 나이대를 확인하라. 
# 일일권의 경우 '일일권'과 '일일권(비회원)'을 모두 포함하라
daily = data[data.대여구분코드.isin(['일일권','일일권(비회원)'])].연령대코드.value_counts().sort_index()
total = data.연령대코드.value_counts().sort_index()
ratio = (daily /total).sort_values(ascending=False)
print(ratio)
print('max ratio age ',ratio.index[0])

# 5. 연령대별 평균 이동거리를 구하여라
result = data.groupby(['연령대코드'])['이동거리'].mean().to_frame('이동거리')
print(result)

# 6. 연령대코드가 20대인 데이터를 추출하고,이동거리값이 추출한 데이터의 이동거리값의 평균 이상인 데이터를 추출한다.
# 최종 추출된 데이터를 대여일자, 대여소번호 순서로 내림차순 정렬 후 1행부터 200행까지의 탄소량의 평균을 
# 소숫점 3째 자리까지 구하여라
con = data[data.연령대코드.isin(['20대'])]
target = con[con.이동거리 >= con.이동거리.mean()]
target = target.sort_values(['대여일자','대여소번호'], ascending = False).reset_index(drop=True).iloc[:200]
target['탄소량'] = target['탄소량'].astype('float')
answer = round(target['탄소량'].sum()/len(target['탄소량']), 3)
print(answer)

# 7. 6월 7일 ~10대의 “이용건수”의 중앙값은?
data['대여일자'] = pd.to_datetime(data['대여일자'])
result = data[(data.연령대코드 == '~10대') & (data.대여일자 == pd.to_datetime('2021-06-07'))].이용건수.median()
print(result)

# 8. 평일 (월~금) 출근 시간대(오전 6,7,8시)의 대여소별 이용 횟수를 구해서 데이터 프레임 형태로 표현한 후
# 각 대여시간별 이용 횟수의 상위 3개 대여소와 이용횟수를 출력하라
target = data[data.day_name.isin(['Tuesday', 'Wednesday', 'Thursday', 'Friday','Monday']) \
              & data.대여시간.isin([6,7,8])]
result = target.groupby(['대여시간','대여소번호']).size().to_frame('이용 횟수')
answer = result.sort_values(['대여시간','이용 횟수'],ascending=False).groupby('대여시간').head(3)
print(answer)

# 9. 이동거리의 평균 이상의 이동거리 값을 가지는 데이터를 추출하여 추출데이터의 이동거리의 표본표준편차 값을 구하여라
mean = data['이동거리'].mean()
target = data[data['이동거리'] >= mean]
answer = target.이동거리.std()
print(answer)

# 10. 남성(‘M’ or ‘m’)과 여성(‘F’ or ‘f’)의 이동거리값의 평균값을 구하여라
data['성별'] = data['성별'].str.upper()
women = data[data.성별.isin(['F'])].이동거리.mean()
men = data[data.성별.isin(['M'])].이동거리.mean()
print('women = ', women, 'men = ', men)

# or 테이블로 구하는 방법
#data['sex'] = data['성별'].map(lambda x: '남' if x in ['M','m'] else '여')
#answer = data[['sex','이동거리']].groupby('sex').mean()
#print(answer)


# 전 세계 행복도 지표 데이터
source = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/happy2/happiness.csv', encoding='utf-8')
print(source.head(3))

# 1. 데이터는 2018년도와 2019년도의 전세계 행복 지수를 표현한다. 
# 각 년도의 행복랭킹 10위를 차지한 나라의 행복점수의 평균을 구하여라

# 10위까지 모든 나라 합산 평균 구할 때 사용. answer처럼 계산도 된다는 거 확인하고 넘어가기
#in_2018 = source[source['년도'] == 2018].reset_index(drop=True)[:10]
#in_2019 = source[source.년도 == 2019].reset_index(drop = True)[:10]
#answer = ((in_2018.점수) + (in_2019.점수)).mean()
#print(answer)

result = source[source.행복랭킹 == 10]['점수'].mean()
print(result)

# 2. 데이터는 2018년도와 2019년도의 전세계 행복 지수를 표현한다. 
# 각 년도의 행복랭킹 50위이내의 나라들의 각각의 행복점수 평균을 데이터프레임으로 표시하라

in_2018 = source[source['년도'] == 2018].sort_values('행복랭킹',ascending = True).reset_index(drop=True)[:50]
in_2019 = source[source.년도 == 2019].sort_values('행복랭킹',ascending = True).reset_index(drop = True)[:50]

mean1 = in_2018.점수.mean()
mean2 = in_2019.점수.mean()

answer1 = pd.DataFrame([mean1, mean2], index = [2018, 2019], columns = ['점수평균'])
answer1.index.name = '년도'
print(answer1)

# or 그냥 데이터프레임에서 뽑아내는 방법
#result = source[source.행복랭킹 <= 50][['년도','점수']].groupby('년도').mean()
#print(result)

# 3. 2018년도 데이터들만 추출하여 행복점수와 부패에 대한 인식에 대한 상관계수를 구하여라
result = source[source.년도 == 2018][['점수','부패에 대한인식']].corr().iloc[0,1]
print(result)

# 4. 2018년도와 2019년도의 행복랭킹이 변화하지 않은 나라명의 수를 구하여라
result = len(source[['행복랭킹','나라명']]) - len(source[['행복랭킹','나라명']].drop_duplicates())
print(result)

# 5. 2019년도 데이터들만 추출하여 각 변수간 상관계수를 구하고 
# 내림차순으로 정렬한 후 상위 5개를 데이터 프레임으로 출력하라. 컬럼명은 v1,v2,corr으로 표시하라

#데이터가 이상한지 출력이 안됨..
#zz = source[source.년도 ==2019].corr().unstack().to_frame().reset_index().dropna()
#result = zz[zz[0] != 1].sort_values(0,ascending=False).drop_duplicates(0)
#answer = result.head(5).reset_index(drop=True)
#answer.columns = ['v1','v2','corr']
#print(answer)

# 6. 각 년도별 하위 행복점수의 하위 5개 국가의 평균 행복점수를 구하여라
for_2018 = source[source.년도 == 2018].sort_values('행복랭킹', ascending = True)[-5:]
for_2019 = source[source.년도 == 2019].sort_values('행복랭킹', ascending = True)[-5:]

pro = pd.concat([for_2018, for_2019], join = 'outer')
answer = pro.groupby('년도')['점수'].mean()
print(answer)

# or
#result = source.groupby('년도').tail(5).groupby('년도').mean()[['점수']]
#print(result)

# 7. 2019년 데이터를 추출하고 
# 해당데이터의 상대GDP 평균 이상의 나라들과 평균 이하의 나라들의 행복점수 평균을 각각 구하고 그 차이값을 출력하라
target = source[source.년도 == 2019]
con = target.상대GDP.mean()

upper = target[target.상대GDP >= con].점수.mean()
lower = target[target.상대GDP <= con].점수.mean()
answer = upper - lower
print(answer)

# 8. 각 년도의 부패에 대한인식을 내림차순 정렬했을때 상위 20개 국가의 부패에 대한인식의 평균을 구하여라
result = source.sort_values(['년도','부패에 대한인식'], ascending = False)\
    .groupby('년도').head(20).groupby(['년도'])[['부패에 대한인식']].mean()
print(result)

# 9. 2018년도 행복랭킹 50위 이내에 포함됐다가 2019년 50위 밖으로 밀려난 국가의 숫자를 구하여라
result = set(source[(source.년도 == 2018) & (source.행복랭킹 <= 50)].나라명)\
      - set(source[(source.년도 == 2019) & (source.행복랭킹 <= 50)].나라명)
answer = len(result)
print(answer)

# 10. 2018년,2019년 모두 기록이 있는 나라들 중 년도별 행복점수가 가장 증가한 나라와 그 증가 수치는?
count = source.나라명.value_counts()
target = count[count >= 2].index

df2 = source.copy()
multiple = df2[df2.나라명.isin(target)].reset_index(drop=True)
multiple.loc[multiple['년도'] == 2018,'점수'] = multiple[multiple.년도 == 2018]['점수'].values * (-1)
result = multiple.groupby('나라명')['점수'].sum().sort_values().to_frame().iloc[-1]
print(result)