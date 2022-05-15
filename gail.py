"""An implementation of GAIL with WGAN discriminator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class Discriminator(tf.keras.Model):
  """Implementation of a discriminator network."""

  def __init__(self, input_dim):
      """Initializes a discriminator.
      Args:
         input_dim: size of the input space.
      """
      super(Discriminator, self).__init__()
      kernel_init = tf.keras.initializers.Orthogonal(gain=1.0)

      self.main = tf.keras.Sequential([
          tf.keras.layers.Dense(
              units = 100,
              input_shape=(input_dim,),
              activation='tanh',
              kernel_initializer=kernel_init),
          tf.keras.layers.Dense(
              units=100, activation='tanh', kernel_initializer=kernel_init),
          tf.keras.layers.Dense(units=1, kernel_initializer=kernel_init)

      ])

  def call(self, inputs):
    """Performs a forward pass given the inputs.
    Args:
      inputs: a batch of observations (tf.Variable).
    Returns:
      Values of observations.
    """
    return tf.clip_by_value(
              self.main(inputs), -1, 1, name=None
          )


class GAIL(object):
  """Implementation of GAIL (https://arxiv.org/abs/1606.03476).
  Instead of the original GAN, it uses WGAN (https://arxiv.org/pdf/1704.00028).
  """

  def __init__(self, input_dim, subsampling_rate, lambd=10.0, gail_loss='airl'):
    """Initializes actor, critic, target networks and optimizers.
    Args:
       input_dim: size of the observation space.
       subsampling_rate: subsampling rate that was used for expert trajectories.
       lambd: gradient penalty coefficient for wgan.
       gail_loss: gail loss to use.
    """

    self.subsampling_rate = subsampling_rate
    self.lambd = lambd
    self.gail_loss = gail_loss
    self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    with tf.compat.v1.variable_scope('discriminator'):
      self.disc_step = tf.Variable(
          0, dtype=tf.int64, name='step')
      self.discriminator = Discriminator(input_dim)
      self.discriminator_optimizer = tf.compat.v1.train.AdamOptimizer()
      self.discriminator_optimizer._create_slots(self.discriminator.variables)  # pylint: disable=protected-access

  def update(self, policy_states,policy_actions,expert_states,expert_actions,dones):
    """Updates the WGAN potential function or GAN discriminator.
    Args:
       batch: A batch from training policy.
       expert_batch: A batch from the expert.
    """
    # obs = tf.Variable(
    #     np.stack(batch.obs).astype('float32'))
    # expert_obs = tf.Variable(
    #     np.stack(expert_batch.obs).astype('float32'))

    # expert_mask = tf.Variable(
    #     np.stack(expert_batch.mask).astype('float32'))

    # Since expert trajectories were resampled but no absorbing state,
    # statistics of the states changes, we need to adjust weights accordingly.
    #expert_mask = tf.maximum(0, -expert_mask)
    expert_weight = dones / self.subsampling_rate + (1 - dones)
    expert_weight = tf.expand_dims(
        expert_weight, axis = 1, name=None
    )

    inputs = tf.concat([policy_states, policy_actions], -1)
    expert_inputs = tf.concat([expert_states, expert_actions], -1)

    # Avoid using tensorflow random functions since it's impossible to get
    # the state of the random number generator used by TensorFlow.
    alpha = np.random.uniform(size=(inputs.get_shape()[0], 1))
    alpha = tf.Variable(alpha.astype('float32'))
    inter = alpha * inputs + (1 - alpha) * expert_inputs

    with tf.GradientTape() as tape:
      output = self.discriminator(inputs)
      expert_output = self.discriminator(expert_inputs)

      # real_loss = self.cross_entropy(tf.ones_like(expert_output), expert_output)
      # fake_loss = self.cross_entropy(tf.zeros_like(output), output)
      loss_on_real = tf.compat.v1.losses.sigmoid_cross_entropy(
          tf.ones_like(expert_output),
          expert_output,
          expert_weight,
          label_smoothing = 0.0,
          scope = None,
          loss_collection=None,
          reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

      loss_on_generated = tf.compat.v1.losses.sigmoid_cross_entropy(
          tf.zeros_like(output),
          output,
          expert_weight,
          scope=None,
          loss_collection=None,
            reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
      # L = - real_weights * log(sigmoid(D(x)))- generated_weights * log(1 - sigmoid(D(G(z))))
      gan_loss = loss_on_real + loss_on_generated
      with tf.GradientTape() as tape2:
        tape2.watch(inter)
        output = self.discriminator(inter)
        grad = tape2.gradient(output, [inter])[0]

      grad_penalty = tf.reduce_mean(input_tensor=tf.pow(tf.norm(tensor=grad, axis=-1) - 1, 2))

      loss = gan_loss + self.lambd * grad_penalty

    grads = tape.gradient(loss, self.discriminator.variables)

    self.discriminator_optimizer.apply_gradients(
        zip(grads, self.discriminator.variables), global_step=self.disc_step)

  def get_reward(self, obs, action):  # pylint: disable=unused-argument
    if self.gail_loss == 'airl':
      inputs = tf.concat([obs, action], -1)
      return self.discriminator(inputs)
    else:
      inputs = tf.concat([obs, action], -1)
      return -tf.math.log(1 - tf.nn.sigmoid(self.discriminator(inputs)) + 1e-8)

  @property
  def variables(self):
    """Returns all variables including optimizer variables.
    Returns:
      A dictionary of all variables that are defined in the model.
      variables.
    """
    disc_vars = (
        self.discriminator.variables + self.discriminator_optimizer.variables()
        + [self.disc_step])

    return disc_vars

# if __name__ == '__main__':
#     g = GAIL(input_dim =300+100,subsampling_rate= 0.01)
#     # action (100
#     t1 = np.random.randn(5,300)
#     t2 = np.random.randn(5,100)
#     p = 0.5
#     t3 = np.random.choice(a=[False, True], size=(5, 1), p=[p, 1-p])
#     t1 = tf.constant(t1,dtype = tf.float32)
#     t2 = tf.constant(t2,dtype = tf.float32)
#     r1 = g.get_reward(t1, t2, next_obs=None)
#     g.update(t1,t2,t1,t2,t3)
#     # inputs = tf.concat([t1, t2],-1)
#     r2 = g.get_reward(t1,t2,next_obs=None)

