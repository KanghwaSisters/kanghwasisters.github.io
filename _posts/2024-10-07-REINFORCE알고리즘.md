---
title: "[RL] REINFORCE알고리즘"
excerpt: "Policy Gradient ,REINFORCE"
categories: 24-2개인발표
tags: 
    - [RL, 이채연]
toc: true
toc_sticky: true
comments: true
use_math: true
author: Chaeyeon Lee
header:
  teaser: ../assets/image/Thumbnail/24_2_Presentation.png

date: 2024-10-07
---

# REINFORCE 알고리즘

2024-10-07 이채연


# 정책 기반 강화 학습

## 가치 기반 강화 학습 vs. 정책 기반 강화학습

|  | 가치 기반 강화 학습 | 정책 기반 강화 학습 |
| --- | --- | --- |
| 개념 | 가치함수에 의해서 각각의 상태마다 행동에 대해서 가치를 판단한다. 이를 통해 현재 상태에 대해 가장 최적의 행동을 찾을 수 있다. | 가치를 찾는 것이 아니므로 가치함수를 사용하지 않고 정책을 직접적으로 최적화한다. |
| 행동 공간 | 이산적 행동
예) 그리드게임: 왼쪽, 오른쪽, 이동, 정지 | 연속적 혹은 확률적 행동
예) 로봇제어: 팔 각도는 연속적(-30도~30도) |
| 예시 알고리즘 | Q-learning, Deep Q-Network(DQN) | REINFORCE |
| 목표 | 가치 함수 최적화 | 정책 함수 최적화 |
| 정책 | 탐욕적 정책 | 확률적 정책 |
| 출력(딥살사) | 선형 | 비선형(softmax) |

## 정책 기반 강화학습의 장단점

### 장점

1. 연속적 행동 공간에서 확률적으로 행동을 선택하기 때문에 선택지에 제한되지 않는다. 따라서 복잡한 환경에서 더 잘 작동할 수 있는 유연성을 가지고 있다고 할 수 있다.
2. 확률적 정책을 사용하기 때문에 탐색과 학습이 자연스럽다. 행동을 확률적으로 선택하면서 다양한 행동을 시도할 수 있기 때문이다.

### 단점

1. 안정적인 학습을 위해 다양한 경험이 필요하고 이에 따라 샘플이 많이 필요하여 학습 비용이 증가한다. 즉, 샘플 효율성(sample efficiency)가 감소한다.  다양한 경험이 필요한 이유는 확률적으로 행동을 선택하기 때문에 같은 상태에서도 다른 행동을 시도할 수 있기 때문이다. 또한, 초기에 불확실하게 행동을 선택하므로 그레이디언트 추정의 분산이 커져 정책 업데이트가 불안정하므로 경험을 통해 분산을 줄여야 하는 것이다.
2. global이 아닌 local에 수렴하는 경우가 있다. 

---

# 폴리시 그레이디언트

**정책신경망**은 정책 기반 강화학습에서 정책을 근사하는 인공신경망을 의미한다. 정책신경망을 사용하는 경우에 정책신경망의 가중치에 따라 에이전트가 받을 누적 보상이 달라진다. 즉, 누적 보상은 정책신경망의 가중치에 따라 결정된다. 

그리드월드에서의 정책은 상태마다 행동에 따라 확률을 나타내는 것이기에 테이블의 형태로 정책을 가지고 있어야 했지만, 정책신경망으로 정책을 대체하면서 θ라는 정책 신경망의 가중치값이 정책으로 표현할 수 있다.

$$
\pi_\theta(a | s)=P[a | s, \theta]
$$

누적 보상은 최적화하고자 하는 목표함수 J(θ)가 되며 최적화를 하게 되는 변수는 인공신경망의 가중치가 된다. 가중치의 값으로 누적보상이 달라지기 때문이다. 정책 기반 강화학습의 목표를 수식으로 나타내면 다음과 같다. 

$$
maximzieJ(\theta)
$$

## 경사상승법(Gradient Ascent)

목표함수 J(θ)를 최적화하는 방법은 목표함수를 미분해서 그 미분값에 따라 정책을 업데이트하는 것으로 목표함수를 최대화하기 위해 경사가 올라가는 **경사상승법**을 이용한다.

$$
\theta_{t+1} = \theta_t + \alpha\nabla_{\theta}J(\theta)
$$

위의 식처럼 목표함수의 경사상승법을 따라서 근사된 정책을 업데이트하는 방식을 **폴리시 그레이디언트**라고 한다. 이때 목표함수의 정의는 다음과 같다.

$$
J(\theta)=v_{{\pi}_{\theta}}(s_0)
$$

만일 에피소드의 끝이 있고 에이전트가 어떤 특정 상태 s0에서 에피소드를 시작하는 경우에 목표함수는 상태 s0에 대한 가치함수라고 할 수 있다. 즉, 앞에서 이야기한 것처럼 누적 보상이 최적화하고자 하는 목표함수 J(θ)가 되고, 정책신경망의 가중치에 따라 누적보상이 달라진다고 했으므로 이러한 가치함수의 형태로 나타나게 된 것이다.

## 폴리시 그레이디언트 정리

하지만 가치함수를 미분하는 것은 어려움이 있기 때문에 **폴리시 그레이티언트 정리**가 필요하다. 파라미터 θ와 성능지표 J(θ)의 관계를 도식화해보면 아래와 같다.

$$
\begin{aligned}
\nabla_{\theta}J(\theta)=\sum_{s}{d_{\pi_\theta}(s)}\sum_{a}{\nabla_{\theta}\pi_{\theta}(a|s)q_{\pi}(s,a)}
&=\sum_{s}{d_{\pi_\theta}}\sum_{a}{\nabla_{\theta}\pi_{\theta}(a|s)\times{\frac{\nabla_{\theta}\pi_{\theta}(a|s)}{\pi_{\theta}(a|s)}}q_{\pi}(s,a)}\\
&=\sum_{s}{d_{\pi_\theta}}\sum_{a}{\nabla_{\theta}\pi_{\theta}(a|s)\times{\nabla_{\theta}log\pi_{\theta}(a|s)}q_{\pi}(s,a)}
\end{aligned}
$$

이때 d는 상태 분포(State Distribution)에서 분포의 d를 따온 변수로 d(s)는 간단히 말해 정책을 따랐을 때 s라는 상태에 에이전트가 있을 확률을 말한다. d(s)를 곱해주는 이유는 에이전트가 방문할 상태들의 중요성을 반영하기 위해서이다. 즉, 에이전트가 자주 방문하는 상태와 거의 방문하지 않는 상태를 동일하게 취급하지 않겠다는 것이다. 위의 수식의 의미는 가능한 모든 상태에 대해 각 상태에서 특정 행동을 했을 때 받을 큐함수의 기댓값의 미분을 의미한다. 식을 정리하여 얻은 최종 수식은 다음과 같다.

$$
\nabla_{\theta}J(\theta)=E_{\pi_{\theta}}[\nabla_{\pi_{\theta}}log\pi_{\theta}(a|s)q_{\pi}(s,a))]
$$

이는 목표함수의 미분값, 즉 경사를 의미한다. 강화학습 알고리즘처럼 폴리시 그레이디언트에도 기댓값은 샘플링으로 대체할 수 있으므로 에이전트가 정책신경망을 업데이트하기 위해 구해야 하는 식은 그 안에 있는 식이다. 이를 통해 아래처럼 폴리시 그레이디언트를 업데이트 할 수 있다. 

$$
\theta_{t+1}=\theta_{t}+\alpha\nabla_{\theta}J(\theta)\approx\theta_{t}+\alpha[\nabla_{\theta}log\pi_{\theta}(a|s)q_{\pi}(s,a)]
$$

---

# REINFORCE 알고리즘

폴리시 그레이디언트에서는 행동을 선택하는 데 가치함수가 필요하지 않다. 따라서 현재 에이전트는 정책만 가지고 있고 가치함수 혹은 큐함수를 가지고 있지 않기 때문에 q(s,a)를 구할 수가 없다. 목표함수의 미분값인 J(θ)를 근사하기 위한 가장 고전적인 방법 중 하나는 큐함수를 반환값 G로 대체하는 것이다. 이처럼 큐함수를 반환값으로 대체하는 것을 **REINFORCE 알고리즘**이라고 한다. 알고리즘의 업데이트 식은 다음과 같다.

$$
\theta_{t+1}=\theta_{t}+\alpha[\nabla_{\theta}log\pi_{\theta}(a|s)G_{t}]
$$

에피소드가 끝날 때까지 기다리면 에피소드 동안 지나온 상태에 대해 각각의 반환값을 구할 수 있다. REINFORCE 알고리즘은 에피소드마다 실제로 얻은 보상으로 학습하는 폴리시 그레이디언트이므로 **몬테카를로 폴리시 그레이디언트**라고도 부른다.

---

# REINFORCE 코드 구현

### 전체 코드

```python
import copy
import pylab
import random
import numpy as np
from environment import Env
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 상태가 입력, 각 행동의 확률이 출력인 인공신경망 생성
class REINFORCE(tf.keras.Model):
    def __init__(self, action_size):
        super(REINFORCE, self).__init__()
        self.fc1 = Dense(24, activation='relu')
        self.fc2 = Dense(24, activation='relu')
        self.fc_out = Dense(action_size, activation='softmax')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        policy = self.fc_out(x)
        return policy

# 그리드월드 예제에서의 REINFORCE 에이전트
class REINFORCEAgent:
    def __init__(self, state_size, action_size):
        # 상태의 크기와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size
        
        # REINFORCE 하이퍼 파라메터
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = REINFORCE(self.action_size)
        self.optimizer = Adam(lr=self.learning_rate)
        self.states, self.actions, self.rewards = [], [], []

    # 정책신경망으로 행동 선택
    def get_action(self, state):
        policy = self.model(state)[0]
        policy = np.array(policy)
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # 반환값 계산
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # 한 에피소드 동안의 상태, 행동, 보상을 저장
    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    # 정책신경망 업데이트
    def train_model(self):
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        
        # 크로스 엔트로피 오류함수 계산
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(model_params)
            policies = self.model(np.array(self.states))
            actions = np.array(self.actions)
            action_prob = tf.reduce_sum(actions * policies, axis=1)
            cross_entropy = - tf.math.log(action_prob + 1e-5)
            loss = tf.reduce_sum(cross_entropy * discounted_rewards)
            entropy = - policies * tf.math.log(policies)

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        self.states, self.actions, self.rewards = [], [], []
        return np.mean(entropy)

if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env(render_speed=0.01)
    state_size = 15
    action_space = [0, 1, 2, 3, 4]
    action_size = len(action_space)
    agent = REINFORCEAgent(state_size, action_size)

    scores, episodes = [], []

    EPISODES = 200
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

            agent.append_sample(state, action, reward)
            score += reward

            state = next_state

            if done:
                # 에피소드마다 정책신경망 업데이트
                entropy = agent.train_model()
                # 에피소드마다 학습 결과 출력
                print("episode: {:3d} | score: {:3d} | entropy: {:.3f}".format(
                      e, score, entropy))

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

## 에이전트와 환경 상호작용

1. 상태에 따른 행동 선택
2. 선택한 행동으로 환경에서 한 타임스텝을 진행
3. 환경으로부터 다음 상태와 보상을 받음
4. 다음 상태에 대한 행동을 선택, 에피소드가 끝날 때까지 반복
5. 환경으로부터 받은 정보를 토대로 에피소드마다 학습을 진행

- 행동 선택
딥살사 에이전트와 달리 REINFORCE 에이전트는 정책신경망을 가지고 있기 때문에 행동을 선택할 때 정책신경망의 출력을 이용하면 된다. 또한, 정책 자체가 확률적이기 때문에 그 확률에 따라 행동을 선택하면 에이전트는 저절로 탐험을 하게 된다.

```python
def get_action(self, state):
        policy = self.model(state)[0]
        policy = np.array(policy)
        return np.random.choice(self.action_size, 1, p=policy)[0]
```

- 정책 신경망 업데이트
큐함수를 에피소드마다 얻는 반환값으로 대체했기 때문에 반환값을 계산하는 함수가 필요하다. 에피소드가 끝나면 에이전트는 그레이디언트를 계산하는데 그 전에 반환값을 계산한다.

에이전트가 타임스텝 6까지 진행하고 에피소드가 끝났을 경우를 예를 들어보자. 반환값은 아래 식처럼 에피소드 동안 지나온 모든 상태에 대해 각각 계산한다.

$$
\begin{aligned}
G_1=R_2+\gamma{R_3}&+\gamma^2{R_4}+\gamma^3{R_5}+\gamma^4{R_6}\\
G_2=R_3+&\gamma{R_2}+\gamma^2{R_2}+\gamma^3{R_6}\\
G_3=R&_4+\gamma{R_5}+\gamma^2{R_6}\\
G_4&=R_5+\gamma{R_6}\\
&G_5=R_6
\end{aligned}
$$

         실제로 코드에서 반환값을 계산할 때는 거꾸로 계산하며 계산해놓은 반환값을 이용해 좀 더 효율적인
         방법을 사용할 수 있다. 

$$
\begin{aligned}
&G_5=R_6\\
G_4&=R_5+\gamma{G_5}\\
G_3&=R_4+\gamma{G_4}\\
G_2&=R_3+\gamma{G_3}\\
G_1&=R_2+\gamma{G_2}\\
\end{aligned}
$$

```python
# 한 에피소드 동안의 상태, 행동, 보상을 저장
def append_sample(self, state, action, reward):
    self.states.append(state[0])
    self.rewards.append(reward)
    act = np.zeros(self.action_size)
    act[action] = 1
    self.actions.append(act)
        
# 반환값 계산
def discount_rewards(self, rewards):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * self.discount_factor + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards
```

- 반환값
계산한 반환값을 이용해 오류함수를 계산할 수 있다. 에이전트는 에피소드가 끝날 때마다 아래 함수를 호출하고 반환값을 구한다. 계산한 반환값을 통해 오류함수를 구하고 그다음 가중치를 얼마다 업데이트할지 구한다.

```python
# 정책신경망 업데이트
def train_model(self):
    # 정책신경망의 업데이트 성능 개선을 위해 정규화
    discounted_rewards = np.float32(self.discount_rewards(self.rewards))
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)
```

- 오류함수
1. 반환값 G가 θ의 함수가 아니기 때문에 그레이디언트의 괄호 안으로 집어넣을 수 있다. 이처럼 변형하면 오류함수가 무엇인지 명확하게 할 수 있다.

$$
\nabla_{\theta}[log\pi_{\theta}(a|s)G_t]
$$

       2. 오류함수의 의미

      * 크로스 엔트로피란?
         크로스 엔트로피는 엔트로피의 변형으로 MSE와 같이 지도학습에서 많이 사용되는 오류함수이다. 아래 
         식이 의미하는 것은 yi와 pi의 값이 얼마나 서로 비슷한가이다. 두 값이 가까워질수록 전체 식의 값은 줄
         어 들어 같아지면서 식의 값은 최소가 된다. 지도학습에서는 y가 보통 정답으로 사용하기 때문에 현재 예
         측 값이 얼마나 정답과 가까운지를 나타내게 되어 오류함수로 사용이 가능하다.

$$
크로스\ 엔트로피=-\sum_iy_ilogp_i
$$

          실제로 선택한 행동을 정답으로 둔 크로스 엔트로피를 통해 정책신경망을 업데이트하면 무조건 실제로
          행동을 더 선택하는 방향으로 업데이트 할 것이다. 그러나 실제로 선택한 행동이 부정적 보상을 받게 했
          다면 그 행동을 선택할 확률을 낮춰야 한다. 따라서 업데이트 값이 행동의 좋고 나쁨의 정보를 가지고 있
          는 반환값과 곱해 정책신경망을 업데이트 한다.

![image.png]({{ site.baseurl }}/assets/image/Articles/2024_2/REINFORCE알고리즘/image1.png)

1. REINFORCE 알고리즘이 정책신경망을 업데이트하는 방식은 경사상승법이다. 

$$
\theta_{t+1}\approx\theta_t+\alpha[\nabla_{\theta}log\pi_\theta(a|s)G_t]=\theta_t-\alpha[\nabla_\theta(-log\pi_\theta(a|s)G_t)]
$$

       그러나 (-)를 붙여 거꾸로 경사를 내려가서 계산해도 결국 똑같은 방향으로 정책신경망은 업데이트 된다.

```jsx
    # 크로스 엔트로피 오류함수 계산
    model_params = self.model.trainable_variables
    with tf.GradientTape() as tape:
        tape.watch(model_params)
        policies = self.model(np.array(self.states))
        actions = np.array(self.actions)
        action_prob = tf.reduce_sum(actions * policies, axis=1)
        # 실제로 선택한 행동을 정답으로 뒀을 때 크로스 엔트로피
        cross_entropy = - tf.math.log(action_prob + 1e-5)
        # 최종 오류함수
        loss = tf.reduce_sum(cross_entropy * discounted_rewards)
        entropy = - policies * tf.math.log(policies)

    # 오류함수를 줄이는 방향으로 모델 업데이트
    grads = tape.gradient(loss, model_params)
    self.optimizer.apply_gradients(zip(grads, model_params))
    self.states, self.actions, self.rewards = [], [], []
    return np.mean(entropy) 
```

# REINFORCE의 실행 결과

![image.png]({{ site.baseurl }}/assets/image/Articles/2024_2/REINFORCE알고리즘/image2.png)

![image.png]({{ site.baseurl }}/assets/image/Articles/2024_2/REINFORCE알고리즘/image3.png)

REINFORCE는 딥살사 에이전트와는 달리 ε-탐욕 정책을 사용하지 않기 때문에 ε에 대한 정보는 출력하지 않는다. 대신 ε-탐욕 정책을 사용하지 않기 때문에 지속적인 탐험을 에이전트가 하기 어렵다. 초반 에이전트는 초록색 세모에 많이 부딪히는데, 이때 에이전트는 초록색 세모에 부딪히지 않도록 학습한다. 따라서 에이전트는 시작점에서 움직이지 않게 되고 목표였던 파란색 동그라미로 갈 방법이 없어진다. 

이러한 문제를 해결하고자 타임스텝마다 (-0.1)의 보상을 에이전트에게 주어 가만히 시작점에 머무는 행동이 좋은 행동이 아닌 것임을 알려주어 에피소드를 끝내줄 파란색 동그라미를 찾아 탐험하게 한다. 그래프를 보면수렴하는 점수의 값이 1이 아닌데 그 이유는 시간이 지날 수록 (-0.1)의 보상을 받기 때문이다. 

![image.png]({{ site.baseurl }}/assets/image/Articles/2024_2/REINFORCE알고리즘/image4.png)

---

출처

1. 파이썬과 케라스로 배우는 강화학습
2. [https://zoomkoding.github.io/강화학습/2019/07/19/RL-2.html](https://with-rl.tistory.com/entry/%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0-%EB%B0%B0%EC%9A%B0%EB%8A%94-%EA%B0%95%ED%99%94-%ED%95%99%EC%8A%B5-09-%EC%A0%95%EC%B1%85-%EA%B8%B0%EB%B0%98-%EC%97%90%EC%9D%B4%EC%A0%84%ED%8A%B8)
3. [https://hiddenbeginner.github.io/rl/2022/09/11/policy_gradient_methods.html](https://hiddenbeginner.github.io/rl/2022/09/11/policy_gradient_methods.html)