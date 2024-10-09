---
title: "[08주차/강화학습 세션/프로젝트intro] 슬롯머신 문제-DQN으로 구현"
excerpt: "슬롯머신 문제,DQN" # 미리보기로 보이는 부분
categories: 24-1강화학습세션
toc: true
toc_sticky: true
comments: true
author: Jimin Lee

date: 2024-05-13

---

# 8주차 세션

## 요약
오랜만의 정규 세션이었던 만큼, 치킨파티를 진행했습니다.  
5주차부터 시작했던 슬롯머신 구현이 시험기간으로 인해 완성이 늦춰져 세션에서 함께 구현했습니다.  
다양한 문제들이 발생하고, 스트레스가 가득한 상황이었지만 포기하지 않고 모든 조원이 코드 구현에 성공했습니다. 🔥🔥🔥

## 주로 발생한 문제
### 오류
- net에 넣는 데이터의 차원이 불일치하는 문제
- done의 boolean 타입이 float/int로 변환되지 않아 발생한 문제
- state애서만 차원을 맞춰주고, next_state에서는 차원을 변경하지 않음
- 어디서부터 tensor 자료형으로 써야하는지 모르겠음.
- target에서 max Q 값을 뽑아내면 pred과 차원이 맞지 않음.
- Net의 input 값의 크기가 맞지 않음.
- tensor([tensor,tensor]) 형태는 작동하지 않음
- test code에서 epsilon을 0으로 만들어줘야 함.

### 개선점
- replay memory 관련 하이퍼파라미터를 정의하지 않음
- reward 값을 명확하게 주자! (쓴 약 : 아무 것도 없음 : 바나나 = 0 : 1 : 2 -> -1 : 0 : 1)
  - *feat. 통계학과답게 표준화시켜서 표현하자 ! - 지민 to 은나*  
- env를 agent의 파라미터로 받아 attributes를 사용하기 
- 충분히 많은 에피소드를 실행시켜봐야 함. 
- 탐험률과 탐험률 감소를 에피소드 길이에 따라 조정해야 한다. 
  - epsilon_min에 닿기 전에 에피소드 iteration이 끝나면 이상함
- action을 string을 혼용하는 것보다, int로 추상화시켜 표현하는 것이 편하다. 
