---
title : "시간차, SARSA, 딥 살사"
Excerpt : "시간차, SARSA, 딥 살사"
categories: 24-2개인발표
tags : [장예원]
Author : Yewon Jang
Date : 2024-09-30
---
# 시간차, SARSA, 딥 살사

@ 장예원_2024년 9월 30일

# 시간차 예측

- **시간차 예측 (Temporal-Difference Prediction)**은 타임스텝마다 가치함수를 업데이트한다. 즉 에이전트는 실시간으로 예측과 현실의 차이로 학습할 수 있다.
- 가치함수의 업데이트는 실시간으로 이뤄지며, 한 번에 하나의 가치함수만 업데이트한다.

$$

V(S_t)←V(S_t)+α(R_{t+1}+γV(S_{t+1})−V(S_t))
$$

- 매 타임스텝마다 에이전트는 **(1)** 현재의 상태 $S_t$에서 행동을 하나 선택하고 **(2)** 환경으로부터 보상 $R_{t+1}$을 받고 **(3)** 다음 상태 $S_{t+1}$를 알게 된다. 에이전트는 현재 가지고 있는 가치함수 리스트에서 다음 상태에 해당하는 가치함수 $V(S_{t+1})$를 가져올 수 있다. 여기서 계산한 $R_{t+1} + γV(S_{t+1})$은 현재 상태  $S_t$의 가치함수 업데이트의 목표가 된다.

                                          $R_{t+1}+γV(S_{t+1})−V(S_t)$ = 업데이트의 목표

                                      $α(R_{t+1}+γV(S_{t+1})−V(S_t))$ = 업데이트의 크기

### **시간차 에러 (Temporal-Difference Error)**

$R_{t+1}+γV(S_{t+1})−V(S_t)$는 **시간차 에러**라고 한다. 시간차 예측에서 업데이트의 목표는 반환값과는 달리 실제의 값은 아니다. $V(S_{t+1})$는 현재 에이전트가 가지고 있는 값인데, 에이전트는 이 값을 $S_{t+1}$의 가치함수일 것이라고 예측하고 있다.

### 부트스트랩 (Bootstrap)

다른 상태의 가치함수 예측값을 통해 지금 상태의 가치함수를 예측하는 방식을 **부트스트랩**이라고 한다. 즉 업데이트 목표도 정확하지 않은 상황에서 가치함수를 업데이트하는 것이다. 

![업데이트 목표.jpg]({{ site.baseurl }}\assets\image\Articles\2024_2\24_2TDP_Sarsa_DeepSarsa\1.jpg)

그림 4.12는 시간차 예측에서 가치함수를 한 번 업데이트하는 과정을 보여준다. 어떤 상태에서 행동을 하면 보상을 받고 다음 상태를 알게 되고 **다음 상태의 가치함수**와 **알게 된 보상**을 더해 그 값을 **업데이트의 목표**로 삼는다는 것이다.

### **시간차 예측의 장점**

1. 에피소드가 끝날 때까지 기다릴 필요 없이 바로 가치함수를 업데이트할 수 있다. 즉 에이전트는 현재 상태에서 행동을 한 번 하고 다음 상태를 알게 되면 바로 이전 상태의 가치함수를 업데이트할 수 있다.
2. 충분히 많은 샘플링을 통해 업데이트하면 많은 경우에 몬테카를로 예측보다 더 효율적으로 빠른 시간 안에 참 가치함수에 근접한다. 

### **시간차 예측의 단점**

몬테카를로 예측보다 초기 가치함수 값에 따라 예측 정확도가 많이 달라진다.

---

# 살사/ 시간차 제어

- **살사 (SARSA)**는 강화학습 알고리즘 흐름을 살펴보면 정책 이터레이션과 가치 이터레이션에서 발전하는데, 살사부터 강화학습이라고 부른다.

![업데이트 목표.jpg]({{ site.baseurl }}\assets\image\Articles\2024_2\24_2TDP_Sarsa_DeepSarsa\2.jpg)

### GPI (Generalized Policy Iteration)

정책 이터레이션은 “정책 평가”와 “정책 발전”을 번갈아 가며 실행하는 과정이다. 벨만 기대 방정식을 이용해 현재의 정책에 대한 참 가치함수를 구하는 것이 **정책 평가**이며, 구한 가치함수에 따라 정책을 업데이트하는 것이 **정책 발전**이다. 이러한 정책 이터레이션을 **GPI**라고 한다. GPI에서는 단 한 번만 정책을 평가해서 가치함수를 업데이트하고 바로 정책을 발전하는 과정을 반복한다.

| **다이내믹 프로그래밍 (GPI)** | **강화 학습 (살사/ 시간차 제어)** |
| --- | --- |
| 정책 평가: 벨만 방정식을 따름 | 정책 평가: 몬테카를로 예측/시간차 예측을 따름 |
| 탐욕 정책 발전:
1. 주어진 가치함수에 대해 새로운 정책을 얻는 과정
2. 모든 상태의 정책을 발전시킴 | 탐욕 정책: 
1. 별도의 정책을 두지 않고 현재 상태에서 가장 큰 가치를 지니는 행동을 선택 
2. 모든 상태의 정책을 발전시킬 수 없음 (타임스텝마다 가치함수를 현재 상태에 대해서만 업데이트함) |
- **살사/ 시간차 제어 (temporal-difference control)** 은 **시간차 예측**과 **탐욕 정책**이 합쳐진 것을 말한다.

![업데이트 목표.jpg]({{ site.baseurl }}\assets\image\Articles\2024_2\24_2TDP_Sarsa_DeepSarsa\3.jpg)

$$
π'(s)=argmax_{a∈A}E_{π}[R_{t+1}+γv_{π}(S_{t+1})|S_{t}=s, A_{t}=a]

$$

현재 상태의 정책을 발전시키려면 argmax 안에 들어있는 값의 최댓값을 알아야 하는데 그러려면 환경의 모델을 알아야 한다. 따라서 위 수식을 시간차 제어의 탐욕 정책으로 사용할 수 없다.

탐욕 정책에서 다음 상태의 가치함수를 보고 판단하는 것이 아니고 현재 상태의 큐함수를 보고 판단한다면 환경의 모델을 몰라도 된다. 시간차 제어에서는 아래 수식으로 표현되는 탐욕 정책을 통해 행동을 선택한다.

### 강화 학습 (살사/시간차 제어)의 탐욕 정책

$$
π(s)=argmax_{a∈A}Q(s,a)
$$

큐함수에 따라 행동을 선택하려면 에이전트는 가치함수가 아닌 큐함수의 정보를 알아야 한다. 따라서 시간차 제어에서는 업데이트하는 대상이 가치함수가 아닌 큐함수가 돼야 한다. 이때 시간차 제어의 식은 아래와 같다. 

### 살사/ 시간차 제어에서 큐함수의 업데이트

$$
Q(S_t,A_t)←Q(S_t,A_t)+α(R_{t+1}+γQ(S_{t+1},A_{t+1})−Q(S_t,A_t))
$$

다음 상태의 큐함수인 $Q(S_{t+1},A_{t+1})$을 알기 위해서는 다음 상태 $S_{t+1}$에서 다음 행동 $A_{t+1}$까지 선택해야 한다.  시간차 제어에서 큐함수를 업데이트하는 것을 그림으로 나타낸 것은 아래와 같다.

![업데이트 목표.jpg]({{ site.baseurl }}\assets\image\Articles\2024_2\24_2TDP_Sarsa_DeepSarsa\4.jpg)

1. 에이전트는 샘플인 상태 $S_t$에서 탐욕 정책에 따라 행동 $A_t$를 선택한다.
2. 환경은 에이전트에게 보상 $R_{t+1}$을 주고 다음 상태 $S_{t+1}$을 알려준다.
3. 한 번 더 에이전트는 탐욕 정책에 따라 행동 $A_{t+1}$을 선택하고 하나의 샘플 $[S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}]$이 생성되면 그 샘플로 큐함수를 업데이트한다.

즉 살사/ 시간차 제어는 **(1)** 현재 가지고 있는 큐함수를 토대로 **(2)** 샘플을 탐욕 정책으로 모으고 **(3)** 그 샘플로 방문한 큐함수를 업데이트하는 과정을 반복하는 것이다.

### ϵ-탐욕 정책

**탐험 (Exploration)**
이미 충분히 많은 경험을 한 에이전트의 경우에는 탐욕 정책이 좋은 선택이겠지만 초기 에이전트에게 탐욕 정책은 잘못된 학습으로 가게 할 가능성이 크다. 따라서 탐욕 정책을 대체할 수 있는 새로운 정책이 필요한데, 그 대안이 **ϵ-탐욕 정책**이다.

![업데이트 목표.jpg]({{ site.baseurl }}\assets\image\Articles\2024_2\24_2TDP_Sarsa_DeepSarsa\5.jpg)

**ϵ-탐욕 정책**은 **1-ϵ만큼의 확률**로는 현재 상태에서 가장 큰 큐함수의 값을 가지는 행동을 선택하지만,  **ϵ만큼의 확률**로는 탐욕적이지 않은 행동을 선택한다. 

현재 가지고 있는 큐함수는 수렴하기 전까지는 편향돼 있는 정확하지 않은 값이다. 따라서 에이전트는 정확하지 않은 큐함수를 토대로 탐욕적으로 행동하기보다는  ϵ인 확률로 검은색 화살표를 따라서 엉뚱한 행동을 한다.

![업데이트 목표.jpg]({{ site.baseurl }}\assets\image\Articles\2024_2\24_2TDP_Sarsa_DeepSarsa\6.jpg)

### ϵ-탐욕 정책의 단점

ϵ-탐욕 정책은 최적의 큐함수를 찾았다 하더라도 ϵ의 확률로 계속 탐험한다는 한계가 있다. 따라서 학습을 진행함에 따라 ϵ의 값을 감소시키는 방법도 사용한다.

**그리드월드 예제에서는 ϵ의 값이 일정한 ϵ-탐욕 정책을 사용한다.*

### 살사/ 시간차 제어 정리

살사/ 시간적 제어는 간단히 **두 단계**로 생각하면 된다.

<aside>

1. ϵ-탐욕 정책을 통해 샘플 $[S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}]$을 획득한다.
2. 획득한 샘플로 다음 식을 통해 큐함수 $Q(S_t, A_t)$를 업데이트한다.
</aside>

→ 큐함수는 에이전트가 가진 정보로서 큐함수의 업데이트는 에이전트 자신을 업데이트하는 것과 같다. 따라서 아래 그림과 같이 큐함수 업데이트를 살사 에이전트를 업데이트한다는 의미에서 화살표로 표현할 수 있다. 

![업데이트 목표.jpg]({{ site.baseurl }}\assets\image\Articles\2024_2\24_2TDP_Sarsa_DeepSarsa\7.jpg)

### 살사/ 시간차 제어 코드 실행 및 결과

```python
import numpy as np
import random
from collections import defaultdict
from environment import Env

class SARSAgent:
    def __init__(self, actions):
        self.actions = actions
        self.step_size = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        # 0을 초기값으로 가지는 큐함수 테이블 생성
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # <s, a, r, s', a'>의 샘플로부터 큐함수를 업데이트
    def learn(self, state, action, reward, next_state, next_action):
        state, next_state = str(state), str(next_state)
        current_q = self.q_table[state][action]
        next_state_q = self.q_table[next_state][next_action]
        td = reward + self.discount_factor * next_state_q - current_q
        new_q = current_q + self.step_size * td
        self.q_table[state][action] = new_q

    # 입실론 탐욕 정책에 따라서 행동을 반환
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 무작위 행동 반환
            action = np.random.choice(self.actions)
        else:
            # 큐함수에 따른 행동 반환
            state = str(state)
            q_list = self.q_table[state]
            action = arg_max(q_list)
        return action

# 큐함수의 값에 따라 최적의 행동을 반환
def arg_max(q_list):
    max_idx_list = np.argwhere(q_list == np.amax(q_list))
    max_idx_list = max_idx_list.flatten().tolist()
    return random.choice(max_idx_list)

if __name__ == "__main__":
    env = Env()
    agent = SARSAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        # 게임 환경과 상태를 초기화
        state = env.reset()
        # 현재 상태에 대한 행동을 선택
        action = agent.get_action(state)

        while True:
            env.render()

            # 행동을 위한 후 다음상태 보상 에피소드의 종료 여부를 받아옴
            next_state, reward, done = env.step(action)
            # 다음 상태에서의 다음 행동 선택
            next_action = agent.get_action(next_state)
            # <s,a,r,s',a'>로 큐함수를 업데이트
            agent.learn(state, action, reward, next_state, next_action)

            state = next_state
            action = next_action

            # 모든 큐함수를 화면에 표시
            env.print_value_all(agent.q_table)

            if done:
                break
```

→ 위 살사/ 시간차 제어 코드를 실행하면 아래와 같은 화면이 나온다. 

![업데이트 목표.jpg]({{ site.baseurl }}\assets\image\Articles\2024_2\24_2TDP_Sarsa_DeepSarsa\8.jpg)

왼쪽 그림에서는 에이전트가 학습을 시작한 지 얼마 되지 않아서 일부 상태만 방문했다. 하지만 시간이 지나면 오른쪽 그림처럼 모든 상태를 방문하게 된다.

1. 다이내믹 프로그래밍에서와 다르게 화면에 버튼이 없다.
2. 모든 상태에 대해 큐함수가 표시되는 것이 아니라 에이전트가 방문한 상태에 대해서만 큐함수를 표시한다. 

---

<aside>

기존의 강화학습 알고리즘에서는 각 상태에 대한 정보를 테이블 형태로 저장했지만 각 상태의 정보를 근사한다면 상태 공간의 크기가 크고 환경이 변해도 학습할 수 있다. 그중에서도 **인공신경망**을 강화학습과 함께 사용하는 움직임이 많다. 

이번 시간에는 인공신경망을 이용해 큐함수를 근사한 **딥살사 (Deep-SARSA)**를 살펴보고 이를 통해 강화학습을 인공신경망과 함께 어떻게 사용하는지 배워보자.

</aside>

# 딥 살사

- **딥 살사 (Deep-SARSA)**는 살사 알고리즘을 사용하되 큐함수를 인공신경망으로 근사한 것이다.
- 은닉층을 두 개 사용할 것이므로 심층신경망이 된다.

그리드월드 문제를 살짝 변형해보자. 그리드월드에서 (1) 장애물의 숫자가 2개 → 3개로 늘고, (2) 이 장애물들이 움직인다면 어떨까? **큐러닝까지의 알고리즘**으로는 변형한 그리드월드 문제를 풀기가 어렵다. 근본적으로 **상태가 적은 문제에만 적용 가능**하기 때문이다. 

변형한 그리드월드와 같은 문제를 해결하려면 테이블이 아닌 근사함수 (Function Approximation) 로 가치함수를 표현해야 한다. 이때 사용할 수 있는 근사함수로는 여러 가지가 있는데, 우리가 다루고자 하는 근사함수는 인공신경망이다.

![업데이트 목표.jpg]({{ site.baseurl }}\assets\image\Articles\2024_2\24_2TDP_Sarsa_DeepSarsa\9.jpg)

다시 앞의 문제로 돌아가자. 에이전트가 장애물을 만날 경우 보상은 (-1)이며 도착했을 경우 보상은 (+1)이다. 에이전트가 해야 할 일은 보상을 최대화하는 것이다. 따라서 에이전트의 목표는 장애물을 피하고 도착지점에 가는 것이 된다.

변형된 그리드월드 예제에서는 MDP의 상태의 정의를 다르게 해야 한다. 에이전트가 장애물을 회피하고 도착지점에 가기에 충분한 정보를 에이전트에게 줘야 한다. ex) 물체의 속도 정보

→ 이 문제에서 정의하는 상태는 다음과 같다.

1. 에이전트에 대한 도착지점의 상대 위치 x, y
2. 도착지점의 라벨
3. 에이전트에 대한 장애물의 상대 위치 x, y
4. 장애물의 라벨
5. 장애물의 속도

| **살사** | **딥살사** |
| --- | --- |
| 큐함수를 테이블 형태로 모든 행동 상태에 대해 업데이트하고 저장 | 큐함수를 인공신경망으로 근사 |
| 하나의 큐함수 값을 업데이트 | 큐함수를 근사하고 있는 인공신경망의 매개변수를 업데이트 (경사하강법을 사용) |

### 딥살사의 오차함수

경사하강법을 사용해 인공신경망을 업데이트하려면 **오차함수**를 정의해야 하는데, 가장 기본적으로 **MSE**를 사용한다. 살사를 이용한 큐함수의 업데이트 식에서 정답의 역할을 하는 것과 예측에 해당하는 것을 알아볼 수 있다.

                                            $R_{t+1}+γQ(S_{t+1},A_{t+1})$ = 정답의 역할을 하는 것

                                                            $Q(S_t,A_t)$ = 예측에 해당하는 것

이 정답과 예측을 MSE 식에 집어넣어서 오차함수를 만들어볼 수 있다. 

$$
MSE=∑(R_{t+1}+γQ_θ(S_{t+1},A_{t+1})−Q_θ(S_t,A_t))^2
$$

여기서 $Q_θ$로 표기하는 것은 테이블 형태의 큐함수가 아니라 **$θ$를 매개변수로 가지는 인공신경망을 통해 표현한 큐함수**라는 뜻이다. 큐함수를 근사하고 있는 인공신경망의 매개변수 $θ$를 이 오차함수를 통해 업데이트하는 것이 학습 과정이다. 

### 딥살사 코드 실행 및 결과

```python
import copy
import pylab
import random
import numpy as np
from environment import Env
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 딥살사 인공신경망
class DeepSARSA(tf.keras.Model):
    def __init__(self, action_size):
        super(DeepSARSA, self).__init__()
        self.fc1 = Dense(30, activation='relu')
        self.fc2 = Dense(30, activation='relu')
        self.fc_out = Dense(action_size)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.fc_out(x)
        return q

# 그리드월드 예제에서의 딥살사 에이전트
class DeepSARSAgent:
    def __init__(self, state_size, action_size):
        # 상태의 크기와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size
        
        # 딥살사 하이퍼 파라메터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.  
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.model = DeepSARSA(self.action_size)
        self.optimizer = Adam(lr=self.learning_rate)

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_values = self.model(state)
            return np.argmax(q_values[0])

    # <s, a, r, s', a'>의 샘플로부터 모델 업데이트
    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 학습 파라메터
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(model_params)
            predict = self.model(state)[0]
            one_hot_action = tf.one_hot([action], self.action_size)
            predict = tf.reduce_sum(one_hot_action * predict, axis=1)

            # done = True 일 경우 에피소드가 끝나서 다음 상태가 없음
            next_q = self.model(next_state)[0][next_action]
            target = reward + (1 - done) * self.discount_factor * next_q

            # MSE 오류 함수 계산
            loss = tf.reduce_mean(tf.square(target - predict))
        
        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))

if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env(render_speed=0.01)
    state_size = 15
    action_space = [0, 1, 2, 3, 4]
    action_size = len(action_space)
    agent = DeepSARSAgent(state_size, action_size)
    
    scores, episodes = [], []

    EPISODES = 1000
    for e in range(EPISODES):
        done = False
        score = 0
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)

            # 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            next_action = agent.get_action(next_state)

            # 샘플로 모델 학습
            agent.train_model(state, action, reward, next_state, 
                                next_action, done)
            score += reward
            state = next_state

            if done:
                # 에피소드마다 학습 결과 출력
                print("episode: {:3d} | score: {:3d} | epsilon: {:.3f}".format(
                      e, score, agent.epsilon))

                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.xlabel("episode")
                pylab.ylabel("score")
                pylab.savefig("./save_graph/graph.png")

        # 100 에피소드마다 모델 저장
        if e % 100 == 0:
            agent.model.save_weights('save_model/model', save_format='tf')
```

→ 위 살사/ 시간차 제어 코드를 실행하면 아래와 같은 화면이 나온다. 

![업데이트 목표.jpg]({{ site.baseurl }}\assets\image\Articles\2024_2\24_2TDP_Sarsa_DeepSarsa\10.jpg)

ϵ-탐욕 정책의 값을 나타내는 epsilon은 에이전트가 얼마만큼의 확신을 가지고 행동을 선택하고 있는지 알려준다. 만약 epsilon이 0.01이라면 0.01의 확률로 탐험을 한다는 것이다.

아래 그림은 100 에피소드를 진행했을 때의 에피소드 점수와 500 에피소드를 진행했을 때의 에피소드 점수의 그래프를 보여준다.

![업데이트 목표.jpg]({{ site.baseurl }}\assets\image\Articles\2024_2\24_2TDP_Sarsa_DeepSarsa\11.jpg)

500 에피소드 정도가 되면 ϵ-탐욕 정책의 ϵ이 0.01이 된다. 따라서 점수가 거의 수렴하는 것을 볼 수 있다.

### ϵ의 감소 속도 중요성

*이 책의 예제에서는 ϵ을 매 타임스텝마다 0.9999를 곱하면서 감소시킨다.

ϵ를 더 빨리 감소시킨다면 점수는 더 빨리 수렴할 수도 있다. 하지만 에이전트가 탐험을 덜 하게 되므로 최적으로 수렴하지 않고 엉뚱한 값으로 수렴할 수도 있다. 

→ 따라서 ϵ을 어느 정도의 속도로 감소시킬지도 정해야 할 변수 중 하나다.