# Winner 공개 예측 앙상블

원본 train 라벨/sample_submission과 공개 submissions-and-uids version 2의 CAT/LGB/XGB best-single test 예측을 사용한다.
모델 재학습 없이 같은 가중치 평균과 XGB 공개 v4→v1 UID PP를 재현하기 위한 실험이다.

`winner_blend.ipynb`를 실행하면 `outputs/released_v2/`에 평균 제출·평균+PP 제출, numpy 예측, metrics/로그를 저장한다.
공개 최종 blend/stack 파일과의 차이를 기록하며, 최종 우승 제출과 완전히 같거나 모델 학습까지 재현했다고 주장하지 않는다.
학습 과정이 없으므로 OOF를 생산하지 않는다. test 정답이 없어 AUC도 계산하지 않는다.

전체 506,691개 test ID에서 실행했다. 평균+PP와 공개 최종 blend의 평균 절대 차이 0.003513, 최대 절대 차이 0.333095였다.
옮긴 PP는 원문 XGB 셀 45와 전체 train/test에서 같은 결과임을 별도로 검증했다.
