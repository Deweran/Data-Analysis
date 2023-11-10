# 문제: 다음은 22명의 학생들이 국어시험에서 받은 점수이다. 학생들의 평균이 75보다 크다고 할 수 있는가?
# H0: 모평균은 mu와 같다. (μ = mu), 학생들의 평균은 75이다
# H1: 모평균은 mu보다 크다. (μ > mu), 학생들의 평균은 75보다 크다
# 가정: 모집단은 정규분포를 따른다, 표본의 크기가 충분히 크다.
# 검정통계량, p-value, 검정결과를 출력하시오

from scipy.stats import ttest_1samp 

# 데이터
scores = [75, 80, 68, 72, 77, 82, 81, 79, 70, 74, 76, 78, 81, 73, 81, 78, 75, 72, 74, 79, 78, 79]

# 모평균 가설검정
mu = 75  # 검정할 모평균
alpha = 0.05  # 유의수준

# t-test를 사용하여 가설 검정
t_statistic, p_value = ttest_1samp(scores, mu, alternative='greater')

# 결과 출력
print("t-statistic:", t_statistic)
print("p-value:", p_value)

if p_value < alpha:
    print("귀무가설을 기각합니다. 모평균은 75보다 큽니다.")
else:
    print("귀무가설을 채택합니다. 모평균은 75보다 크지 않습니다.")