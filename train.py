import argparse
import os
import random

import gym
from gym.wrappers import TimeLimit
import numpy as np
import pandas as pd
import yaml

from env_aug import AntEnvAug, HalfCheetahEnvAug, HopperEnvAug, fixedSwimmerEnv
from model import EnsembleGymEnv
from ppo import PPO, Memory
from train_funcs import (collect_data, test_agent, train_agent,
                         train_agent_model_free, train_agent_model_free_debug)
from online_learning import ExpWeights
from utils import MeanStdevFilter, reward_func, weights_init

os.environ['KMP_DUPLICATE_LIB_OK']='True'


### Main PPO Loop
def train_ppo(params):    
    ## random rollouts
    params['zeros'] = False
    
    b = ExpWeights()
    
    if params['env_name'] == 'HalfCheetah-v2':
        env = HalfCheetahEnvAug()
    elif params['env_name'] == 'Ant-v2':
        env = AntEnvAug()
    elif params['env_name'] == 'Swimmer-v2':
        env = fixedSwimmerEnv()
    elif params['env_name'] == 'Hopper-v2':
        env = HopperEnvAug()
    else:
        raise Exception('Environment not supported')

    env = TimeLimit(env, params['steps'])

    params['ob_dim'] = env.observation_space.shape[0]
    params['ac_dim'] = env.action_space.shape[0]
    params['is_done_func'] = None

    if hasattr(env, 'is_done_func'):
        params['is_done_func'] = env.is_done_func

    env = EnsembleGymEnv(params, env)
       
    # TODO: put these into argparse/separate yaml files

    ############## Hyperparameters ##############
    log_interval = 100          # print avg reward in the interval
    policy_iters = params['policy_iters']        # max training episodes
    ep_steps = params['steps']        # max timesteps in one episode
    
    update_timestep = params['update_timestep']      # update policy every n timesteps
    action_std = 0.5           # constant std for action distribution (Multivariate Normal)
    K_epochs = 10                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    #############################################
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    n_parallel = int(update_timestep / ep_steps)

    env_resets = []
    env_resets_real = []

    env.real_env.seed(params['seed'])
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    for _ in range(n_parallel):
        env_resets.append(env.real_env.reset())
        env_resets_real.append(env.real_env.unwrapped.state_vector())
    
    env_resets = np.array(env_resets)
    env_resets_real = np.array(env_resets_real)

    memory = Memory()
    ppo = PPO(params['seed'], state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    if params['var_max']:
        params['filename'] += '-var-max'
        ppo_rew = PPO(params['seed'], state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)

    if params['model_free']:
        train_agent_model_free(ppo, env, memory, update_timestep, params['seed'], 500, ep_steps, env_resets, env_resets_real)
        return

    iterations = 0

    print("\nCollecting random rollouts...")
    total_timesteps = 0
    timesteps, error = collect_data(params, ppo, memory, env)
    total_timesteps += timesteps
    samples = [total_timesteps]
    rewards, rewards_m, lambdas, errors = [], [], [], []
    b.error_buffer.append(error) # initial baseline
    if params['adapt']:
        params['lam'] = 0
        
    while total_timesteps < params['max_timesteps']:
        
        ## train model
        print("\nTraining Model...")
        env.train_model(max_epochs=params['model_epochs'])
    
        ## train policy in model
        print("\nTraining PPO Policy in Model...")
        ppo.change_policy_std(action_std)

        if params['var_max']:
            train_agent(ppo, env, policy_iters, ep_steps, memory, update_timestep, env_resets, 500, 1, n_parallel, params['var_type'])
            train_agent(ppo_rew, env, policy_iters, ep_steps, memory, update_timestep, env_resets, 500, 0, n_parallel, params['var_type'])
        else:
            train_agent(ppo, env, policy_iters, ep_steps, memory, update_timestep, env_resets, 500, params['lam'], n_parallel, params['var_type'])

        ## test policy in the env
        subset_resets_idx = np.random.randint(0, n_parallel, 10)
        subset_resets = env_resets[subset_resets_idx]
        subset_resets_real = env_resets_real[subset_resets_idx]
        if params['var_max']:
            reward_model = test_agent(ppo_rew, env, memory, ep_steps, subset_resets, subset_resets_real, use_model=True)
            reward_actual = test_agent(ppo_rew, env, memory, ep_steps, subset_resets, subset_resets_real, use_model=False)
        else:
            reward_model = test_agent(ppo, env, memory, ep_steps, subset_resets, subset_resets_real, use_model=True)
            reward_actual = test_agent(ppo, env, memory, ep_steps, subset_resets, subset_resets_real, use_model=False)
        print("\nSamples: %s, Reward in WM: %s, True Reward: %s" %(total_timesteps, np.round(reward_model,4), np.round(reward_actual, 4)))

        ## log progress to file
        rewards.append(reward_actual)
        rewards_m.append(reward_model)
        errors.append(error)
        lambdas.append(params['lam'])
        df = pd.DataFrame({'Samples': samples, 'Reward': rewards, 'Reward_WM': rewards_m, 'Lambdas': lambdas, 'MSEs': errors})
        lam = ['Adaptive' if params['adapt']==1 else 'fixed{}'.format(str(params['lam']))][0]
        save_name = "{}_{}_resid{}_{}_{}".format(params['env_name'], lam, str(params['pca']), params['filename'], str(params['seed']))
        if params['comment']:
            save_name = save_name + '_' + params['comment']
        save_name += '.csv'
        df.to_csv(save_name)

        ## collect more data with the new policy
        print("\nCollecting more data with the new policy...")
        timesteps, error = collect_data(params, ppo, memory, env)
        total_timesteps += timesteps
        samples.append(total_timesteps)
        b.update_dists(error, env.model.valid_loss)
        if params['adapt']:
            params['lam'] = b.sample()
            print("\n Using Lambda = {}".format(str(params['lam'])))

        iterations += 1


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')   ## only works properly for HalfCheetah and Ant
    parser.add_argument('--seed', '-se', type=int, default=0)
    parser.add_argument('--num_models', '-nm', type=int, default=3)
    parser.add_argument('--adapt', '-ad', type=int, default=0)              ## set to 1 for adaptive
    parser.add_argument('--steps', '-s', type=int, default=100)             ## maximum time we step through an env per episode
    parser.add_argument('--outer_steps', '-in', type=int, default=3000)     ## how many time steps/samples we collect each outer loop (including initially)
    parser.add_argument('--max_timesteps', '-maxt', type=int, default=1e8)    ## total number of timesteps
    parser.add_argument('--model_epochs', '-me', type=int, default=2000)    ## max number of times we improve model
    parser.add_argument('--update_timestep', '-ut', type=int, default=50000) ## for PPO only; how many steps to accumulate before training on them
    parser.add_argument('--policy_iters', '-it', type=int, default=2000)    ## max number of times we improve policy
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1)  
    parser.add_argument('--lam', '-la', type=float, default=0)              
    parser.add_argument('--pca', '-pc', type=float, default=0)              ## threshold for residual to stop, try [1e-4,2-e4]
    parser.add_argument('--sigma', '-si', type=float, default=0.01)
    parser.add_argument('--filename', '-f', type=str, default='ModelBased')
    parser.add_argument('--dir', '-d', type=str, default='data')
    parser.add_argument('--yaml_file', '-yml', type=str, default=None)
    parser.add_argument('--uuid', '-id', type=str, default=None)
    parser.add_argument('--fix_std', dest='fix_std', action='store_true')
    parser.add_argument('--var_type', type=str, default='reward', choices=('reward', 'state'))
    parser.add_argument('--model_free', dest='model_free', action='store_true')
    parser.add_argument('--var_max', dest='var_max', action='store_true')
    parser.add_argument('--comment', '-c', type=str, default=None)
    parser.set_defaults(fix_std=False)
    parser.set_defaults(model_free=False)
    parser.set_defaults(var_max=False)

    args = parser.parse_args()
    params = vars(args)

    if params['yaml_file']:
        with open(args.yaml_file, 'r') as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            for config in yaml_config['args']:
                if config in params:
                    params[config] = yaml_config['args'][config]
    
    if not(os.path.exists(params['dir'])):
        os.makedirs(params['dir'])
    os.chdir(params['dir'])
    
    if params['uuid']:
        if not(os.path.exists(params['uuid'])):
            os.makedirs(params['uuid'])
        os.chdir(params['uuid'])
    
    train_ppo(params)
    
if __name__ == '__main__':
    main()
