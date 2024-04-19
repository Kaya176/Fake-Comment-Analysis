# LIWC를 활용한 거짓의견 분석 프로젝트
## 개요
본 프로젝트는 [1]"Oh, Yu Won, and Chong Hyun Park. "Machine cleaning of online opinion spam: Developing a machine-learning algorithm for detecting deceptive comments." American behavioral scientist 65.2 (2021): 389-403." 의 후속 연구로써 인터넷에서 작성되는 글 중 봇이 작성한 텍스트 혹은 거짓 정보를 포함하고 있는 텍스트(이하 거짓의견)와 일반 유저가 작성한 진실성 있는 텍스트(이하 진짜의견) 간 언어학적 차이를 분석하는 프로젝트입니다.

## Data  
전체 데이터는 총 8개의 issue로 구성되어있으며, 각 issue에는 가짜의견과 진짜의견이 구분된 텍스트 데이터가 포함되어 있습니다. 본 프로젝트에서 사용한 데이터는 전체 issue중 하나의 이슈에서 소수의 샘플링한 데이터를 사용하였습니다. 샘플 데이터 중 거짓 의견(Fake comment)은 113개 진짜 의견(True comment)은 108개로 구성되어 있으며, 한국어 텍스트로 구성되어 있습니다. 데이터 수집 및 전처리 방법은 이전 연구논문[1]을 따릅니다.

## Analysis  
수집한 텍스트를 언어학적으로 분석하기 위해 LIWC(Linguistic Inquiry and Word Count) Tool을 이용하였습니다. 이 tool은 word-level 수준에서 단어의 품사와 의미 그리고 감정분석 등을 할 수 있습니다. 하지만, LIWC는 English만 지원하기 때문에, 한국어 텍스트 데이터를 처리하기 위해서 Google Translate API를 사용하여 영어로 번역 후 분석을 진행하였습니다.(참고사항 : LIWC 분석 결과 table에 텍스트도 포함되기 때문에, github repository의 data는 LIWC 분석 결과만 포함되어 있습니다.)

가짜의견과 진짜의견 각각을 독립적으로 LIWC tool을 이용하여 분석하였으며, 총 2개의 테이블이 분석 결과로 나오게 되었으며 각각 (113x120),(108x120) 크기의 column으로 구성된 2개의 table이 분석 결과로 나오게 됩니다 (size = (113,108) x 2 x 120). 분석 목표는 가짜의견과 진짜의견이 각 column에서 얼마나 연관성이 있는지 분석하여, 각 의견마다 상대적으로 연관성이 큰 column 요소를 찾는 것 입니다. 분석을 위해 PCA 및 naive bayes등의 머신러닝 방법도 사용하였으며,특히 본 jupyter notebook에서는 [2]"Shin, Youjin, and Simon S. Woo. "What is in your password? analyzing memorable and secure passwords using a tensor decomposition." The World Wide Web Conference. 2019." 논문에서 제시한 Parafac2 decomposition을 이용한 분석 방법을 구현하였습니다.

구현은 크게 (1) Remove Zero Variance (2) Feature Normalization (3) Parafac2 Decomposition을 이용한 분석으로 3가지 파트로 구성됩니다. 이 중 전처리에 해당하는 부분은 (1) 과 (2) 이며, 분석 part는 (3)입니다. 각 파트에 대해 간단히 정리하면 다음과 같습니다.

**(1) Remove Zero Variance**  
분산이 작은 경우에는 분석할만한 데이터가 적다고 판단하여 해당 column은 분석에서 제외하였습니다.

**(2) Feature Normalization**  
각 column마다 scale이 다르기 때문에 Normalization을 적용하였습니다. 이때 사용한 기법은 Min-Max Normalization을 사용하였습니다.

**(3) Parafac2 Decomposition을 이용한 분석**  
Tensorly에서 제공하는 Parafac2 클래스를 이용하여 decomposition을 수행하였습니다. 각 거짓의견과 진짜의견을 하나의 matrix로 합친 이후에 decomposition을 수행하였습니다. 수행 방법은 참고한 [2] 논문과 동일하게 구현하였습니다. original matrix와 tensor decomposition 이후 다시 reconstruction한 결과 거의 비슷해야 한다고 생각하여 Reconstruction ratio와 Reconstruction error 값을 각 rank마다 구하여 최적의 rank값을 설정하였습니다. 이후 설정한 rank를 이용하여 decomposition을 수행하였습니다. 분해된 텐서를 이용하여 가짜의견과 진짜의견 간 strength를 구하였으며, 이때 지표는 Euclidean distance를 이용하였습니다.

## Result
가짜의견과 진짜의견간 strength를 수치적으로 구한 결과를 이용하여 결과해석을 진행합니다. strength가 작을수록 거리가 가깝다는 의미이므로 더욱 강한 연관성이 있다고 해석할 수 있습니다. 두 집간간의 strength 차이를 이용하여 평균과 분산을 구하여 일정 threshold 이상인 차이가 나는 column에 대해서 유의미한 차이가 있다고 판단하였습니다. 본 프로젝트에서는 적은 수의 샘플 데이터를 이용하여 구현하였기 때문에, 실제 데이터와 차이가 있을 수 있습니다. 본 분석 방법을 활용하여 각 issue 별로 언어학적 차이를 분석할 수 있으며, 전체 issue에 대해 공통적으로 나타나는 특징에 대해서도 분석할 수 있습니다. 

## 더욱 자세한 설명
- 프로젝트와 관련된 자세한 설명(notion) : https://www.notion.so/Fake-Comment-3d1b3ff5fcbe4adca5a2b3799af3e4c6?pvs=4  
- 참고한 논문[2] 정리 글(개인 Blog) : https://kaya-dev.tistory.com/19

## Reference(Paper Link)
(1) [Oh, Yu Won, and Chong Hyun Park. "Machine cleaning of online opinion spam: Developing a machine-learning algorithm for detecting deceptive comments." American behavioral scientist 65.2 (2021): 389-403.](https://journals.sagepub.com/doi/full/10.1177/0002764219878238?casa_token=oqNGE2BLSukAAAAA%3AbeFzWucrLvMTl5oGCNZYIdf_TqWK98dWQ0OpGge7icJgj2V10lNDOFrY-QMTSgGFcJ4ydUZ-0t8v)  
(2)[Shin, Youjin, and Simon S. Woo. "What is in your password? analyzing memorable and secure passwords using a tensor decomposition.](https://dl.acm.org/doi/abs/10.1145/3308558.3313690?casa_token=oa4an3hr5w8AAAAA:noo8AjmaBU_Ppu9NyoZbeT2LKDu6bRAFUO2QTs6CtG_DbAi1LiXgkvbPx8EyTEjc6SYNUO3ntpB2)