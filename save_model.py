from network import Actor
actor = Actor(100, 128,1e-3, 10, 0.001)
actor.build_networks()
actor.load_weights('./model/epoch_9000/ddpg_actor.h5')


#actor.network.save('mymodel1')
#model = keras.models.load_model('D:\\projects\\airl_rs_tf\\model\\epoch_8000\\ddpg_actor.h5')