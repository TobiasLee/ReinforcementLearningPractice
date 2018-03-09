
import numpy as np
import tensorflow as tf


np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(self,
                 n_actions,
                 n_features,
                 lr=0.01,
                 reward_decay=0.95,
                 output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = lr
        self.gamma = reward_decay
        self.sess = tf.Session()

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self._build_net()  # ？
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        # input
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features]) #
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name='actions_num')
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="action_valuie")
        # fc1
        layer = tf.layers.dense(inputs=self.tf_obs,
                                units=10,
                                activation=tf.nn.tanh,
                                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                bias_initializer=tf.constant_initializer(0.1),
                                name = 'fc1')
        # fc 2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2')

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')
        with tf.name_scope("loss"):
            # to maximize total reward log p * R = minimize - log_p *R
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)

        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={
            self.tf_obs: observation[np.newaxis, :]
        })
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel()) # selction action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_as.append(a)
        self.ep_rs.append(r)
        self.ep_obs.append(s)

    def _discount_and_norm_reward(self): # reward decay and normalize
        discount_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discount_ep_rs[t] = running_add

        # normalize episode rewards
        # discount_ep_rs -= np.mean(discount_ep_rs)
        # discount_ep_rs /= np.std(discount_ep_rs)
        return discount_ep_rs

    def learn(self):
        discount_ep_rs_norm = self._discount_and_norm_reward()

        # train on episode
        self.sess.run(
            self.train_op, feed_dict= {
                self.tf_obs: np.vstack(self.ep_obs), # [None, n_obs]
                self.tf_acts: np.array(self.ep_as), # [None, ]
                self.tf_vt: discount_ep_rs_norm
            }
        )

        self.ep_obs.clear()
        self.ep_rs.clear()
        self.ep_as.clear()
        return discount_ep_rs_norm
