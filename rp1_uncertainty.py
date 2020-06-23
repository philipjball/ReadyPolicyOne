import argparse
import os
import random

import gym
from gym.wrappers import TimeLimit
import numpy as np
import pandas as pd
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
import GPy
from copy import deepcopy
from scipy.stats import spearmanr

from env_aug import AntEnvAug, HalfCheetahEnvAug, HopperEnvAug, fixedSwimmerEnv
from model import EnsembleGymEnv, Transition, TransitionDataset, Model
from ppo import PPO, Memory
from online_learning import ExpWeights
from utils import MeanStdevFilter, reward_func, weights_init

os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_samples(params, ppo, memory, ensemble_env, train=1):
    rollouts = []
    timesteps = 0
    env = ensemble_env.real_env
    while timesteps < 5000:
        if train:
            new_std = 1
        else:
            new_std = np.random.uniform(0,3)
        ppo.change_policy_std(new_std)
        rollout = []
        done = False
        state = env.reset()
        ensemble_env.state_filter.update(state)
        newdata = []
        while (not done):
            action = ppo.select_action(ensemble_env.state_filter(state), memory)
            newdata.append(np.concatenate((state, action)))
            nextstate, reward, done, _ = env.step(action)
            rollout.append(Transition(state, action, reward, nextstate))
            state = nextstate
            if train:
                ensemble_env.state_filter.update(state)
                ensemble_env.action_filter.update(action)
            timesteps += 1
        rollouts.append(rollout)
    for rollout in rollouts:
        if train:
            ensemble_env.model.add_data(rollout)
            ensemble_env.update_diff_filter()
        else:
            ensemble_env.model.add_data_validation(rollout)
            
def swimmer_reward(nextstate, action):
    reward = (nextstate[-1] - 0.0001 * np.linalg.norm(action **2, ord=1))
    return reward

def getvars(model, model_type):
    samples = env.model.memory_val.sample(5000)
    transition_loader = DataLoader(
                TransitionDataset(samples, model.state_filter, model.action_filter, model.diff_filter),
                shuffle=True,
                batch_size=1,
                pin_memory=True
                )
    diff_mean = torch.FloatTensor(model.diff_filter.mean).to(device)
    diff_stddev = torch.FloatTensor(model.diff_filter.stdev).to(device)
    variances = []
    for x_batch, state_batch, nextstate_batch, r_batch in transition_loader:
        x_batch, state_batch = x_batch.to(device, non_blocking=True), state_batch.to(device, non_blocking=True)
        if model_type == 'ensemble':
            preds = []
            for i in model.model.models:
                m = model.model.models[i]
                y_pred, _ = m.forward(x_batch)
                nextstate_pred = (y_pred * diff_stddev) + diff_mean + state_batch
                state_pred = nextstate_pred.detach().numpy()[0]
                action = x_batch.detach().numpy()[0][-2:]
                reward = swimmer_reward(state_pred, action)
                preds.append(reward)
            variances.append(np.std(preds))
    return(variances)

def makeGPdataset(env):
    samples = env.model.memory.sample(5000)
    transition_loader = DataLoader(
                TransitionDataset(samples, model.state_filter, model.action_filter, model.diff_filter),
                shuffle=True,
                batch_size=50000,
                pin_memory=True
                )
    diff_mean = torch.FloatTensor(model.diff_filter.mean).to(device)
    diff_stddev = torch.FloatTensor(model.diff_filter.stdev).to(device)
    for x_batch, state_batch, nextstate_batch, r_batch in transition_loader:
        x_batch, nextstate_batch = x_batch.to(device, non_blocking=True), nextstate_batch.to(device, non_blocking=True)
    X = x_batch.detach().numpy()
    y = nextstate_batch.detach().numpy()
    return(X, y)

def getXtest(env):
    samples = env.model.memory_val.sample(5000)
    transition_loader = DataLoader(
                TransitionDataset(samples, model.state_filter, model.action_filter, model.diff_filter),
                shuffle=True,
                batch_size=50000,
                pin_memory=True
                )
    diff_mean = torch.FloatTensor(model.diff_filter.mean).to(device)
    diff_stddev = torch.FloatTensor(model.diff_filter.stdev).to(device)
    for x_batch, state_batch, nextstate_batch, r_batch in transition_loader:
        x_batch = x_batch.to(device, non_blocking=True)
    X = x_batch.detach().numpy()
    return(X)    
        

### Main PPO Loop
def train_ppo(params):    
    ## random rollouts
    params['zeros'] = False
    
    b = ExpWeights()
    
    env = fixedSwimmerEnv()

    env = TimeLimit(env, params['steps'])

    params['ob_dim'] = env.observation_space.shape[0]
    params['ac_dim'] = env.action_space.shape[0]
    params['is_done_func'] = None

    if hasattr(env, 'is_done_func'):
        params['is_done_func'] = env.is_done_func

    params['num_models'] = 5
    env = EnsembleGymEnv(params)
    
       
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

    # collect data
    get_samples(params, ppo, memory, env, train=1)
    get_samples(params, ppo, memory, env, train=0)
  
    env.train_model(max_epochs=params['model_epochs'])
    ensemble_5 = getvars(env, model_type = 'ensemble')
    
    params['num_models'] = 20
    env20 = deepcopy(env)
    env20.model.models = {i:Model(input_dim=params['ob_dim'] + params['ac_dim'],
                               output_dim=params['ob_dim'],
                               is_done_func = params['is_done_func'],
                               seed = params['seed'] + i,
                               num=i) 
                        for i in range(params['num_models'])}
    env20.train_model(max_epochs=params['model_epochs'])
    ensemble_20 = getvars(env20, model_type = 'ensemble')
    
    
    X, y = makeGPdataset(env)
    kernel = GPy.kern.RBF(input_dim=X.shape[1], variance=1., lengthscale=1.)
    m = GPy.models.GPRegression(X, y, kernel)
    m.optimize(messages=True)
    
    Xtest = getXtest(env)
    gp_preds = m.predict(Xtest)
    
    corr = spearmanr(ensemble_5, ensemble_20)[0]
    print("Correlation = {}".format(str(np.round(corr, 2))))
    
    corr = spearmanr(ensemble_5, gp_preds[1])[0]
    print("Correlation = {}".format(str(np.round(corr, 2))))    
    
    


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Swimmer-v2')   ## only works properly for HalfCheetah and Ant
    parser.add_argument('--seed', '-se', type=int, default=0)
    parser.add_argument('--num_models', '-nm', type=int, default=5)
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
    parser.add_argument('--comment', '-c', type=str, default=None)

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






