#from airl_ddpg import AIRL_DDPG
from ddpg import DDPG,DDPGExpert
from airl import AIRL
#from sac import SAC,SACExpert

ALGOS = {
    #'airl': AIRL_DDPG,
    'airl':AIRL,
    'DDPG': DDPG,
    'DDPGExpert': DDPGExpert,
    # 'PPO': PPO,
    # 'PPOExpert': PPOExpert,
    #'SAC': SAC,
    #'SACExpert':SACExpert
}