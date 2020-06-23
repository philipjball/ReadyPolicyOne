import os

import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)

def reward_func(s1, s2, a, env_name, state_filter=None, is_done_func=None):
    if state_filter:
        s1_real = s1 * state_filter.stdev + state_filter.mean
        s2_real = s2 * state_filter.stdev + state_filter.mean
    else:
        s1_real = s1
        s2_real = s2
    if env_name == "HalfCheetah-v2":
        return np.squeeze(s2_real)[-1] - 0.1 * np.square(a).sum()
    if env_name == "Ant-v2":
        if is_done_func:
            if is_done_func(torch.Tensor(s2_real).reshape(1,-1)):
                return 0.0
        return np.squeeze(s2_real)[-1] - 0.5 * np.square(a).sum() + 1.0
    if env_name == "Swimmer-v2":
        return np.squeeze(s2_real)[-1] - 0.0001 * np.square(a).sum()
    if env_name == "Hopper-v2":
        if is_done_func:
            if is_done_func(torch.Tensor(s2_real).reshape(1,-1)):
                return 0.0
        return np.squeeze(s2_real)[-1] - 0.1 * np.square(a).sum() - 3.0 * np.square(s2_real[0] - 1.3) + 1.0


class MeanStdevFilter():
    def __init__(self, shape, clip=3.0):
        self.eps = 1e-4
        self.shape = shape
        self.clip = clip
        self._count = 0
        self._running_sum = np.zeros(shape)
        self._running_sum_sq = np.zeros(shape) + self.eps
        self.mean = 0
        self.stdev = self.eps

    def update(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1,-1)
        self._running_sum += np.sum(x, axis=0)
        self._running_sum_sq += np.sum(np.square(x), axis=0)
        # assume 2D data
        self._count += x.shape[0]
        self.mean = self._running_sum / self._count
        self.stdev = np.sqrt(
            np.maximum(
                self._running_sum_sq / self._count - self.mean**2,
                 self.eps
                 ))
    
    def __call__(self, x):
        return np.clip(((x - self.mean) / self.stdev), -self.clip, self.clip)

    def invert(self, x):
        return (x * self.stdev) + self.mean


def tidy_up_weight_dir(guids=None):
    if guids == None:
        guids = []
    files = [i for i in os.listdir("./data/") if i.endswith("pth")]
    for weight_full in files:
        weight = weight_full.split('_')[1]
        if weight.split('.')[0] not in guids:
            os.remove("./data/" + weight_full)

def prepare_data(state, action, nextstate, state_filter, action_filter, diff_filter):
    state_filtered = state_filter(state)
    action_filtered = action_filter(action)
    state_action_filtered = np.concatenate((state_filtered, action_filtered), axis=1)
    delta = np.array(nextstate) - np.array(state)
    delta_filtered = diff_filter(delta)
    return state_action_filtered, delta_filtered


def get_residual(newdata, pca_data, pct=0.99):
    X_pca = np.array(pca_data)
    # standardize
    X_pca = (X_pca - np.mean(X_pca)) / (np.std(X_pca) + 1e-8)
    
    Q, Sigma, _ = np.linalg.svd(X_pca.T)
    # proportion
    weight = np.cumsum(Sigma / np.sum(Sigma))
    index = np.sum((weight > pct) == 0)
    train_resid = 1-weight[index]
    V = Q[:,:index+1]
    
    basis = V.dot(V.T)
    
    X = np.array(newdata)
    # standardize with respect to old data
    X = (X - np.mean(X_pca)) / (np.std(X_pca) + 1e-8)
    orig = X.T.dot(X)
    projected = np.matmul(np.matmul(basis, orig), basis)
    residual = (np.trace(orig) - np.trace(projected))/np.trace(orig)
    return(residual, train_resid)


def get_stats(env, state_action_filtered, state_f, action, diff_mean, diff_stddev, state_mean, state_stddev, done, dynamics=False):
    with torch.no_grad():
        stats = []
        for model in env.model.models.values():
            diff_filtered = model.forward(state_action_filtered)
            nextstate_f = state_f + filter_torch_invert(diff_filtered, diff_mean, diff_stddev)
            nextstate = filter_torch_invert(nextstate_f, state_mean, state_stddev)
            if dynamics:
                stats.append(nextstate_f)
            else:
                reward = torch_reward(env.name, nextstate, action, done)
                stats.append(reward)
        if dynamics:
            return (torch.stack(stats) - torch.stack(stats).mean((0))).pow(2).sum(2).mean(0).detach().cpu().numpy()
        return np.std(stats, axis=0)


def random_env_forward(data, env):
    """Randomly allocate the data through the different dynamics models"""
    y = torch.zeros((data.shape[0], env.observation_space.shape[0]), device=device)
    allocation = torch.randint(0, len(env.model.models), (data.shape[0],))
    for i in env.model.models:
        data_i = data[allocation == i]
        y_i = env.model.models[i].forward(data_i)
        y[allocation == i] = y_i
    return y


def filter_torch(x, mean, stddev):
    x_f = (x - mean) / stddev
    return torch.clamp(x_f, -3, 3)


def filter_torch_invert(x_f, mean, stddev):
    x = (x_f * stddev) + mean
    return x


def halfcheetah_reward(nextstate, action):
    return (nextstate[:,-1] - 0.1 * torch.sum(torch.pow(action, 2), 1)).detach().cpu().numpy()


def ant_reward(nextstate, action, dones):
    reward = (nextstate[:,-1] - 0.5 * torch.sum(torch.pow(action, 2), 1) + 1.0).detach().cpu().numpy()
    reward[dones] = 0.0
    return reward


def swimmer_reward(nextstate, action):
    reward = (nextstate[:,-1] - 0.0001 * torch.sum(torch.pow(action, 2), 1)).detach().cpu().numpy()
    return reward


def hopper_reward(nextstate, action, dones):
    reward = (nextstate[:,-1] - 0.1 * torch.sum(torch.pow(action, 2), 1) - 3.0 * (nextstate[:,0] - 1.3).pow(2) + 1.0).detach().cpu().numpy()
    reward[dones] = 0.0
    return reward


def torch_reward(env_name, nextstate, action, dones=None):
    if env_name == "HalfCheetah-v2":
        return halfcheetah_reward(nextstate, action)
    elif env_name == "Ant-v2":
        return ant_reward(nextstate, action, dones)
    elif env_name == "Hopper-v2":
        return hopper_reward(nextstate, action, dones)
    elif env_name == "Swimmer-v2":
        return swimmer_reward(nextstate, action)
    else:
        raise Exception('Environment not supported')
