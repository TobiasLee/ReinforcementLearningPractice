import gym
import tensorflow as tf
import numpy as np
from Actor_Critic import *

np.random.seed(2)
tf.set_random_seed(2)

MAX_EPISODE = 300
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 2000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.99     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.05     # learning rate for critic

env = gym.make("CartPole-v0")
env.seed(1)
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

sess = tf.Session()
actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, gamma=GAMMA)

sess.run(tf.global_variables_initializer())


# env.monitor.start("./tmp/exp1")
for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = list()
    while True:
        if RENDER: env.render()
        a = actor.choose_action(s)
        s_, r, done, info = env.step(a)
        if done:
            r = -20
        track_r.append(r)
        # critic learn td_error
        td_error = critic.learn_state(s, r, s_)
        # actor update behave
        actor.learn(s, a, td_error)

        s = s_ #update state
        t += 1

        if done:
            # if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)
            if "running_reward" not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum*0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True
            print("episode :", i_episode, " reward: ", int(running_reward))
            break

