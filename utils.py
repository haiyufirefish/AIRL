import numpy as np
import random

# td target
def calculate_td_target(discount_factor, rewards, q_values, dones):

    y_t = np.copy(q_values)
    for i in range(q_values.shape[0]):
        y_t[i] = rewards[i] + (1 - dones[i]) * (discount_factor * q_values[i])
    return y_t

# match state and embeddings
def get_label_batch(dict, batch_ids, embedding_dim):

    size = len(batch_ids)
    states = np.zeros((size, 3 * embedding_dim), dtype=np.float32)
    actions = np.zeros((size, embedding_dim), dtype=np.float32)
    dones = np.zeros(size, bool)
    batch_ids = batch_ids.flatten().tolist()
    for id,index in zip(batch_ids, range(size)):
        if(id not in dict):
            print(batch_ids)
            print(id)

        states[index],actions[index],dones[index] = random.choice(dict[id])

    return states,actions,dones
#
