#from airl_ddpg import AIRL_DDPG
from ddpg import DDPG,DDPGExpert
from airl import AIRL
from ppo import PPO
from sac import SAC,SACExpert

ALGOS = {
    #'airl': AIRL_DDPG,
    'airl':AIRL,
    'ddpg': DDPG,
    'ddpgexpert': DDPGExpert,
     'ppo': PPO,
    # 'PPOExpert': PPOExpert,
     'sac': SAC,
    #'SACExpert':SACExpert
}