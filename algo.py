from airl_ddpg import AIRL_DDPG
from ddpg import DDPG,DDPGExpert
#from sac import SAC,SACExpert

ALGOS = {
    'airl': AIRL_DDPG,
    'DDPG': DDPG,
    'DDPGExpert': DDPGExpert,
    # 'PPO': PPO,
    # 'PPOExpert': PPOExpert,
    #'SAC': SAC,
    #'SACExpert':SACExpert
}