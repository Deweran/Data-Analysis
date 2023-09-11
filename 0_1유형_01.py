import pandas as pd
import numpy as np

# 유튜브 인기동영상 데이터
dataurl = 'https://raw.githubusercontent.com/Datamanim/datarepo/main/youtube/youtube.csv'
data = pd.read_csv(dataurl, index_col=0)

# 1. 인기동영상 제작횟수가 많은 채널 상위 10개명을 출력하라 (날짜기준, 중복포함)
con = data.channelId.isin(data.channelId.value_counts().head(10).index)
answer = list(data[con].channelTitle.unique())
print(answer)

# 2. 논란으로 인기동영상이 된 케이스를 확인하고 싶다. 
# dislikes수가 like 수보다 높은 동영상을 제작한 채널을 모두 출력하라
con = data['dislikes'] > data['likes']
ans = list(data[con]['channelTitle'].unique())
print(ans)

# 3. 채널명을 바꾼 케이스가 있는지 확인하고 싶다. 
# channelId의 경우 고유값이므로 이를 통해 채널명을 한번이라도 바꾼 채널의 갯수를 구하여라
change = data[['channelTitle','channelId']].drop_duplicates().channelId.value_counts()
print(change.head()) # 중복털어내고 고유값의 갯수 세기
target = change[change>1]
print(len(target))

# 4. 일요일에 인기있었던 영상들 중 가장많은 영상 종류(categoryId)는 무엇인가?
data['trending_date2'] = pd.to_datetime(data['trending_date2'])
con = data['trending_date2'].dt.day_name() == 'Sunday'
answer = data[con]['categoryId'].value_counts().index[0]
print(answer)

# 5. 각 요일별 인기 영상들의 categoryId는 각각 몇개 씩인지 하나의 데이터 프레임으로 표현하라
group = data.groupby([data['trending_date2'].dt.day_name(),'categoryId'],as_index=False).size()
answer= group.pivot(index='categoryId',columns='trending_date2')
print(answer)

# 6. 댓글의 수로 (comment_count) 영상 반응에 대한 판단을 할 수 있다.
# viewcount대비 댓글수가 가장 높은 영상을 확인하라 (view_count값이 0인 경우는 제외한다)
target2 = data[data.view_count != 0]
t = target2.copy()
t['ratio'] = (target2['comment_count']/target2['view_count']).dropna()
result = t.sort_values(by='ratio', ascending=False).iloc[0].title
print(result)

# 7. 댓글의 수로 (comment_count) 영상 반응에 대한 판단을 할 수 있다.
# viewcount대비 댓글수가 가장 낮은 영상을 확인하라 (view_counts, ratio값이 0인경우는 제외한다.)
ratio = (data['comment_count'] / data['view_count']).dropna().sort_values()
ratio[ratio!=0].index[0]

result= data.iloc[ratio[ratio!=0].index[0]].title
print(result)

# 8. like 대비 dislike의 수가 가장 적은 영상은 무엇인가? (like, dislike 값이 0인경우는 제외한다)
con1 = data['dislikes'] != 0
con2 = data['likes'] != 0
target = data[con1 & con2]
num = (target['dislikes']/target['likes']).sort_values().index[0]

answer = data.iloc[num].title
print(answer)

# 9. 가장많은 트렌드 영상을 제작한 채널의 이름은 무엇인가? (날짜기준, 중복포함)
answer = data.loc[data.channelId == data.channelId.value_counts().index[0]].\
    channelTitle.unique()[0]
print(answer)

# 10. 20회(20일)이상 인기동영상 리스트에 포함된 동영상의 숫자는?
answer= (data[['title','channelId']].value_counts()>=20).sum()
print(answer)
