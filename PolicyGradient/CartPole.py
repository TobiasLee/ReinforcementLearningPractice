import gym
from RL_agent import PolicyGradient
import matplotlib.pyplot as plt

threshold = 400
render = False

env = gym.make("CartPole-v0")
env.seed(1)
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions = env.action_space.n,
    n_features = env.observation_space.shape[0],
    lr= 0.01,
    reward_decay = 0.99
)

for i in range(100):
    # print(i)
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        RL.store_transition(observation, action, reward)

        if done:
            ep_rs_sum = sum(RL.ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum
            if running_reward > threshold:
                render = True  # rendering
            print("episode:", i, "  reward:", int(running_reward))

            vt = RL.learn()
            if i == 0:
                plt.plot(vt)
                plt.xlabel('episode steps')
                plt.ylabel('normalized reward values')
                plt.show()
            # print("Before break")
            break
            # print("After break")

        observation = observation_


observation = env.reset()
render = True
while True:
    # if render:
    env.render()
    action = RL.choose_action(observation)
    observation_, reward, done, info = env.step(action)
    RL.store_transition(observation, action, reward)
    if done:
        print("Game over")
        break
    observation = observation_
