import pandas as pd
import numpy as np 

# 유튜브 공범 컨텐츠 동영상 데이터 
channel = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/youtube/channelInfo.csv')
video = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/youtube/videoInfo.csv')
print(channel.head(2))
print(video.head(2))

# 1. 각 데이터의 ‘ct’컬럼을 시간으로 인식할수 있게 datatype을 변경하고 video 데이터의 videoname의 
# 각 value 마다 몇개의 데이터씩 가지고 있는지 확인하라

video['ct'] = pd.to_datetime(video['ct'])
ans = video.groupby('videoname').size()
# or ans = video['videoname'].value_counts()
print(ans)

# 2. 수집된 각 video의 가장 최신화 된 날짜의 viewcount값을 출력하라
ans = video.sort_values(['videoname', 'ct']).drop_duplicates('videoname', keep = 'last')\
    [['viewcnt','videoname','ct']].reset_index(drop=True)
print(ans)

# 3. Channel 데이터중 2021-10-03일 이후 각 채널의 처음 기록 됐던 구독자 수(subcnt)를 출력하라
channel['ct'] = pd.to_datetime(channel['ct'])
target = channel[channel.ct >= pd.to_datetime('2021-10-03')].sort_values(['ct','channelname'])\
    .drop_duplicates('channelname')
answer = target[['channelname','subcnt']].reset_index(drop = True)
print(answer)

# 4. 각채널의 2021-10-03 03:00:00 ~ 2021-11-01 15:00:00 까지 구독자수 (subcnt) 의 증가량을 구하여라
end = channel.loc[channel.ct.dt.strftime('%Y-%m-%d %H') =='2021-11-01 15']
start = channel.loc[channel.ct.dt.strftime('%Y-%m-%d %H') =='2021-10-03 03']
# dt.strftime => convert 'datetime' data type to 'object' date type 

end_df = end[['channelname','subcnt']].reset_index(drop=True)
start_df = start[['channelname','subcnt']].reset_index(drop=True)

end_df.columns = ['channelname','end_sub']
start_df.columns = ['channelname','start_sub']

tt = pd.merge(start_df,end_df)
tt['del'] = tt['end_sub'] - tt['start_sub']
result = tt[['channelname','del']]
print(result)

# 5. 각 비디오는 10분 간격으로 구독자수, 좋아요, 싫어요수, 댓글수가 수집된것으로 알려졌다. 
# 공범 EP1의 비디오정보 데이터중 수집간격이 5분 이하, 20분이상인 데이터 구간(해당 시점 전,후)의 
# 시각을 모두 출력하라
import datetime 

ep_one = video.loc[video.videoname.str.contains('1')].sort_values('ct').reset_index(drop=True)

ep_one[
        (ep_one.ct.diff(1) >= datetime.timedelta(minutes=20)) | \
        (ep_one.ct.diff(1) <= datetime.timedelta(minutes=5))
      ]

answer = ep_one[ep_one.index.isin([720,721,722,723,1635,1636,1637])]
print(answer)

# 6. 각 에피소드의 시작날짜(년-월-일)를 에피소드 이름과 묶어 데이터 프레임으로 만들고 출력하라
start_date = video.sort_values(['ct','videoname']).drop_duplicates('videoname')[['ct','videoname']]
start_date['date'] = start_date.ct.dt.date
answer = start_date[['date','videoname']]
print(answer)

# 7. “공범” 컨텐츠의 경우 19:00시에 공개 되는것으로 알려져있다. 
# 공개된 날의 21시의 viewcnt, ct, videoname 으로 구성된 데이터 프레임을 viewcnt를 내림차순으로 정렬하여 출력하라
video['time']= video.ct.dt.hour

answer = video.loc[video['time'] == 21] \
            .sort_values(['videoname','ct'])\
            .drop_duplicates('videoname') \
            .sort_values('viewcnt',ascending = False)[['videoname','viewcnt','ct']]\
            .reset_index(drop=True)

print(answer)

# 8. video 정보의 가장 최근 데이터들에서 각 에피소드의 싫어요/좋아요 비율을 ratio 컬럼으로 만들고 
# videoname, ratio로 구성된 데이터 프레임을 ratio를 오름차순으로 정렬하라
one = video.sort_values(['ct','videoname']).drop_duplicates('videoname', keep = 'last')
one['ratio'] = one['dislikecnt'] / one['likecnt']
answer = one[['videoname','ratio']].sort_values('ratio', ascending = True).reset_index(drop = True)
print(answer)

# 9. 2021-11-01 00:00:00 ~ 15:00:00까지 각 에피소드별 viewcnt의 증가량을 데이터 프레임으로 만드시오
start = pd.to_datetime("2021-11-01 00:00:00")
end = pd.to_datetime("2021-11-01 15:00:00")

target = video.loc[(video["ct"] >= start) & (video['ct'] <= end)].reset_index(drop=True)

def check(x):
    result = max(x) - min(x)
    return result

answer = target[['videoname','viewcnt']].groupby("videoname").agg(check)
print(answer)

# 10. video 데이터 중에서 중복되는 데이터가 존재한다. 중복되는 각 데이터의 시간대와 videoname 을 구하여라
answer  = video[video.index.isin(set(video.index) -  set(video.drop_duplicates().index))]
# 초기 인덱스 집합에서 중복값이 두 개 모두 사라진 집합을 빼면 중복값인 인덱스만 남을 것 
result = answer[['videoname','ct']]
print(result)

# 월드컵 출전선수 골기록 데이터 
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/worldcup/worldcupgoals.csv')
print(data.head(3))

# 1. 주어진 전체 기간의 각 나라별 골득점수 상위 5개 국가와 그 득점수를 데이터프레임형태로 출력하라
target = data[['Country','Goals']].groupby(['Country']).sum().sort_values('Goals', ascending = False).head()
print(target)

# 2. 주어진 전체기간동안 골득점을 한 선수가 가장 많은 나라 상위 5개 국가와 
# 그 선수 숫자를 데이터 프레임 형식으로 출력하라
result = data.groupby(['Country']).size().sort_values(ascending = False).head()
print(result)

# 3. Years 컬럼은 년도 -년도 형식으로 구성되어있고, 각 년도는 4자리 숫자이다. 
# 년도 표기가 4자리 숫자로 안된 케이스가 존재한다. 해당 건은 몇건인지 출력하라
data['yearLst'] = data.Years.str.split('-')
print(data['yearLst'].head(3))

def checkFour(x):
    for value in x:
        if len(str(value)) != 4:
            return False
        
    return True
    
data['check'] = data['yearLst'].apply(checkFour)

result = len(data[data.check == False])
print(result)

# 4. Q3에서 발생한 예외 케이스를 제외한 데이터프레임을 df2라고 정의하고 데이터의 행의 숫자를 출력하라 
# (아래 문제부터는 df2로 풀이하겠습니다)
data2 = data[data['check'] == True].reset_index(drop = True)
print(data2.shape)
print(data2.shape[0])

# 5. 월드컵 출전횟수를 나타내는 ‘LenCup’ 컬럼을 추가하고 4회 출전한 선수의 숫자를 구하여라
data2['LenCup'] = data2['yearLst'].str.len()
print(data2['yearLst'].str.len())
# result = data2['LenCup'].value_counts()[4] or
result = len(data2[data2['LenCup'] == 4])
print(result)

# 6. Yugoslavia 국가의 월드컵 출전횟수가 2회인 선수들의 숫자를 구하여라
con1 = data2['Country'] == 'Yugoslavia'
con2 = data2['LenCup'] == 2
result = len(data2[con1 & con2])
print(result)

# 7. 2002년도에 출전한 전체 선수는 몇명인가?
result = len(data2[data2.Years.str.contains('2002')])
print(result)

# 8. 이름에 ‘carlos’ 단어가 들어가는 선수의 숫자는 몇 명인가? (대, 소문자 구분 x)
result = len(data2[data2['Player'].str.lower().str.contains('carlos')])
print(result)

# 9. 월드컵 출전 횟수가 1회뿐인 선수들 중에서 가장 많은 득점을 올렸던 선수는 누구인가?
con = data2['LenCup'] == 1
result = data2[con].sort_values('Goals', ascending = False).reset_index(drop = True).loc[0,'Player']
print(result)

# 10. 월드컵 출전횟수가 1회 뿐인 선수들이 가장 많은 국가는 어디인가?
con = data2['LenCup'] == 1
answer = data2[con].groupby(['Country']).size().sort_values(ascending = False).index[0]
print(answer)

# or 
# result= data2[data2.LenCup==1].Country.value_counts().index[0]
# print(result)
