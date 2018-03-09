import numpy as np
import tensorflow as tf


class Actor:
    def __init__(self, sess, n_features, n_actions, lr=1e-3):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], name='state')
        self.a = tf.placeholder(tf.int32, None, name="act")
        self.td_error = tf.placeholder(tf.float32, None, name="tf_error")

        # build layer
        with tf.variable_scope("Actor"):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(.1),
                name="layer1"
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(.1),
                name="softmax"
            )
        with tf.variable_scope("exp_v"):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # log * advantage
        with tf.variable_scope("train"):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {
            self.s: s,
            self.a: a,
            self.td_error: td
        }
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s): # choose action based on the satate
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel(),)


class Critic:
    def __init__(self, sess, n_features, lr=1e-2, gamma=0.99):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, "reward")

        with tf.variable_scope("Critic"):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(.1),
                name ='l1'
            )
            self.v = tf.layers.dense(
                inputs=l1,
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(.1),
                name='V'
            )
        with tf.variable_scope("td_square_error"):
            self.td_error = self.r + gamma * self.v_ - self.v
            self.loss = tf.square(self.td_error)
        with tf.variable_scope("train"):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)


    def learn_state(self, s, r, s_):
        # advantage function A = r + V_t+1 - V_t
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run(
            [self.td_error, self.train_op],
            {self.s: s,self.r: r,self.v_: v_})
        return td_error
