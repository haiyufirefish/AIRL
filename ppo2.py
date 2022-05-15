# import wandb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
import numpy as np
import os

from base import Algorithm

#



class Actor:
    def __init__(self, state_dim, action_dim, action_bound, std_bound, actor_lr, clip_ratio):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = std_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(actor_lr)
        self.clip_ratio = clip_ratio
        tf.keras.backend.set_floatx('float64')
    def get_action(self, state):
        #print(state)
        # state = np.reshape(state, [1, self.state_dim])
        #print(state)
        mu, std = self.model.predict(state)
        #action  = tf.random.normal([1,self.action_dim],mu,std)
        action = np.random.normal(mu[0], std[0], size=self.action_dim)
        action = np.clip(action, -self.action_bound, self.action_bound)
        log_policy = self.log_pdf(mu, std, action)

        return log_policy, action

    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / \
                         var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def create_model(self):
        state_input = Input((self.state_dim,))
        dense_1 = tf.keras.layers.Dense(128, activation='relu')(state_input)
        dense_2 = Dense(128, activation='relu')(dense_1)
        out_mu = Dense(self.action_dim, activation='tanh')(dense_2)
        mu_output = Lambda(lambda x: x * self.action_bound)(out_mu)
        std_output = Dense(self.action_dim, activation='softplus')(dense_2)
        return tf.keras.models.Model(state_input, [mu_output, std_output])

    def compute_loss(self, log_old_policy, log_new_policy, actions, gaes):
        ratio = tf.exp(log_new_policy - tf.stop_gradient(log_old_policy))
        gaes = tf.stop_gradient(gaes)
        clipped_ratio = tf.clip_by_value(
            ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
        return tf.reduce_mean(surrogate)

    def train(self, log_old_policy, states, actions, gaes):
        with tf.GradientTape() as tape:
            mu, std = self.model(states, training=True)
            log_new_policy = self.log_pdf(mu, std, actions)
            loss = self.compute_loss(
                log_old_policy, log_new_policy, actions, gaes)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def load_weights(self, path):
        self.model.load_weights(path)

    def save_weights(self, path):
        self.model.save_weights(path)


class Critic:
    def __init__(self, state_dim, critic_lr):
        self.state_dim = state_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(critic_lr)

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')
        ])

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape[0] == td_targets.shape[0]
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def load_weights(self, path):
        self.model.load_weights(path)

    def save_weights(self, path):
        self.model.save_weights(path)


class PPO2(Algorithm):
    def __init__(self, state_dim, action_shape, seed, env, gamma=0.9, clip_ratio=0.2, lamdb=0.95,
                 actor_lr=3e-4, critic_lr=1e-3,update_interval = 10,update_epochs = 5):
        super().__init__(10, action_shape, seed, gamma)

        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_shape
        self.action_bound = 1
        self.std_bound = [1e-2, 1.0]
        self.name = 'ppo2'
        self.lamdb = lamdb
        self.update_interval = update_interval
        self.update_epochs = update_epochs
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_opt = tf.keras.optimizers.Adam(actor_lr)
        self.critic_opt = tf.keras.optimizers.Adam(critic_lr)
        self.actor = Actor(self.state_dim, self.action_dim,
                           self.action_bound, self.std_bound, actor_lr, clip_ratio)
        self.critic = Critic(self.state_dim, critic_lr)

    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + self.gamma * forward_val - v_values[k]
            gae_cumulative = self.gamma * self.lamdb * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        return gae, n_step_targets

    def update(self, epochs):

        states, actions, rewards, old_policys = self.buffer.sample()
        next_state, reward, done, _ = self.env.step(actions)
        v_values = self.critic.model.predict(states)
        next_v_value = self.critic.model.predict(next_state)
        gaes, td_targets = self.gae_target(rewards, v_values, next_v_value, done)

        for epoch in range(epochs):
            actor_loss = self.actor.train(
                old_policys, states, actions, gaes)
            critic_loss = self.critic.train(states, td_targets)

    def load_weights(self, path):
        print('load weights success!')
        self.actor.load_weights('{}/ppo_actor.h5'.format(path))
        self.critic.load_weights('{}/ppo_critic.h5'.format(path))

    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.actor.save_weights('{}/ppo_actor.h5'.format(save_dir))
        self.critic.save_weights('{}/ppo_critic.h5'.format(save_dir))

    @property
    def networks(self):
        return [
            self.actor.model,
            self.critic.model,
        ]
if __name__ == '__main__':
    gamma = 0.99
    clip_ratio = 0.2
    lamdb = 0.95
    action_bound = 1
    std_bound = [1e-2, 1.0]
    actor = Actor(300, 100,action_bound, std_bound, 1e-3, clip_ratio)
    s = tf.random.uniform(shape=[1,300])
    p,a = actor.get_action(s)
    print(a.shape)
    save_dir = './model'
    actor.save_weights('{}/ppo_actor.h5'.format(save_dir))
    actor.save_weights('{}/ppo_actor.h5'.format(save_dir))
