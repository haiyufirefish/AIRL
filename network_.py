import tensorflow as tf
import numpy as np


class ActorNetwork(tf.keras.Model):
    def __init__(self, embedding_dim, hidden_dim):
        super(ActorNetwork, self).__init__()
        # print(embedding_dim,"em dim")
        # print(hidden_dim,"hid dim")
        self.inputs = tf.keras.layers.InputLayer(name='input_layer', input_shape=(3 * embedding_dim,))
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu', name='fc1'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(hidden_dim, activation='relu', name='fc2'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(embedding_dim, activation='tanh', name='fc3')
        ])

    def call(self, x):
        x = self.inputs(x)
        return self.fc(x)


class Actor(object):

    def __init__(self, embedding_dim, hidden_dim, learning_rate, state_size, tau):
        self.embedding_dim = embedding_dim
        self.state_size = state_size

        # actor network / target network
        self.network = ActorNetwork(embedding_dim, hidden_dim)
        self.target_network = ActorNetwork(embedding_dim, hidden_dim)
        #  optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        # soft target network update hyperparameter
        self.tau = tau

    def build_networks(self):
        #  Build networks
        self.network(np.zeros((1, 3 * self.embedding_dim)))
        self.target_network(np.zeros((1, 3 * self.embedding_dim)))

    def update_target_network(self):
        #  soft target network update target:evaluation c:policy
        c_theta, t_theta = self.network.get_weights(), self.target_network.get_weights()
        for i in range(len(c_theta)):
            t_theta[i] = self.tau * c_theta[i] + (1 - self.tau) * t_theta[i]
        self.target_network.set_weights(t_theta)

    def train(self, states, dq_das):
        with tf.GradientTape() as g:
            outputs = self.network(states)
            # loss = outputs*dq_das

        dj_dtheta = g.gradient(outputs, self.network.trainable_weights, -dq_das)
        grads = zip(dj_dtheta, self.network.trainable_weights)
        self.optimizer.apply_gradients(grads)

    def save_weights(self, path):
        self.network.save_weights(path)

    def load_weights(self, path):
        print("load successful")
        self.network.load_weights(path)

class CriticNetwork(tf.keras.Model):
    def __init__(self, embedding_dim, hidden_dim):
        super(CriticNetwork, self).__init__()
        self.inputs = tf.keras.layers.InputLayer(input_shape=(embedding_dim, 3 * embedding_dim))
        self.fc1 = tf.keras.layers.Dense(embedding_dim, activation='relu')
        self.concat = tf.keras.layers.Concatenate()
        self.n1 = tf.keras.layers.BatchNormalization()
        self.fc2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.n2 = tf.keras.layers.BatchNormalization()
        self.fc3 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.out = tf.keras.layers.Dense(1, activation='linear')

    def call(self, x):
        s = self.fc1(x[1])
        s = self.concat([x[0], s])
        s = self.n1(s)
        s = self.fc2(s)
        s = self.n2(s)
        s = self.fc3(s)
        return self.out(s)


class Critic(object):

    def __init__(self, hidden_dim, learning_rate, embedding_dim, tau):
        self.embedding_dim = embedding_dim

        # critic network / target network
        self.network = CriticNetwork(embedding_dim, hidden_dim)
        self.target_network = CriticNetwork(embedding_dim, hidden_dim)
        # pptimiz erq
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # MSE
        self.loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        # soft target network update hyperparameter
        self.tau = tau

    def build_networks(self):
        # DQN
        self.network([np.zeros((1, self.embedding_dim)), np.zeros((1, 3 * self.embedding_dim))])
        self.target_network([np.zeros((1, self.embedding_dim)), np.zeros((1, 3 * self.embedding_dim))])
        self.network.compile(self.optimizer, self.loss)

    def update_target_network(self):
        c_omega = self.network.get_weights()
        t_omega = self.target_network.get_weights()
        for i in range(len(c_omega)):
            t_omega[i] = self.tau * c_omega[i] + (1 - self.tau) * t_omega[i]
        self.target_network.set_weights(t_omega)

    def dq_da(self, inputs):
        actions = inputs[0]
        states = inputs[1]
        with tf.GradientTape() as g:
            actions = tf.convert_to_tensor(actions)
            g.watch(actions)
            outputs = self.network([actions, states])
        q_grads = g.gradient(outputs, actions)
        return q_grads

    def train(self, inputs, td_targets, weight_batch):
        weight_batch = tf.convert_to_tensor(weight_batch, dtype=tf.float32)
        with tf.GradientTape() as g:
            outputs = self.network(inputs)
            loss = self.loss(td_targets, outputs)
            weighted_loss = tf.reduce_mean(loss * weight_batch)
        dl_domega = g.gradient(weighted_loss, self.network.trainable_weights)
        grads = zip(dl_domega, self.network.trainable_weights)
        self.optimizer.apply_gradients(grads)
        return weighted_loss

    def train(self, inputs, td_targets):
        with tf.GradientTape() as g:
            outputs = self.network(inputs)
            loss = self.loss(td_targets, outputs)
        dl_domega = g.gradient(loss, self.network.trainable_weights)
        grads = zip(dl_domega, self.network.trainable_weights)
        self.optimizer.apply_gradients(grads)
        return tf.reduce_mean(loss)


    def save_weights(self, path):
        self.network.save_weights(path)

    def load_weights(self, path):
        self.network.load_weights(path)