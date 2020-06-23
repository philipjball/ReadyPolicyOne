import itertools
import random
from collections import deque, namedtuple
from typing import List

import gym
from gym.wrappers import TimeLimit
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from env_aug import HalfCheetahEnvAug, AntEnvAug, HopperEnvAug, fixedSwimmerEnv
from utils import MeanStdevFilter, prepare_data, reward_func

### Model

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'nextstate'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EnsembleGymEnv(gym.Env):
    """Wraps the Ensemble with a gym API, Outputs Normal states, and contains a copy of the true environment"""
    def __init__(self, params, env):
        super(EnsembleGymEnv, self).__init__()
        self.name = params['env_name']
        self.real_env = env
        self.model = Ensemble(params)
        self.observation_space = self.real_env.observation_space
        self.action_space = self.real_env.action_space
        self.state_filter = MeanStdevFilter(self.observation_space.shape[0])
        self.action_filter = MeanStdevFilter(self.action_space.shape[0])
        self.diff_filter = MeanStdevFilter(self.observation_space.shape[0])
        self.current_state = self.reset()
        self.reward_func = reward_func
        self.action_bounds = self.get_action_bounds()
        self.spec = self.real_env.spec
        self._elapsed_steps = 0
        self._max_timesteps = params['steps']
        torch.manual_seed(params['seed'])

    def get_action_bounds(self):
        Bounds = namedtuple('Bounds', ('lowerbound', 'upperbound'))
        lb = self.real_env.action_space.low
        ub = self.real_env.action_space.high
        return Bounds(lowerbound=lb, upperbound=ub)

    def seed(self, seed=None):
        return self.real_env.seed(seed)

    def update_diff_filter(self):
        """To get an unbiased estimate of the difference statistics"""
        self.diff_filter = MeanStdevFilter(self.observation_space.shape[0])
        if len(self.model.memory) != 0:
            train_data = self.model.memory.get_all()
            diff = self.state_filter(train_data.nextstate) - self.state_filter(train_data.state)
            self.diff_filter.update(diff)
        if len(self.model.memory_val) !=0:
            validation_data = self.model.memory_val.get_all()
            diff_val = self.state_filter(validation_data.nextstate) - self.state_filter(validation_data.state)
            self.diff_filter.update(diff_val)

    def train_model(self, max_epochs, n_samples: int = 200000):
        self.model.train_model(
            self.state_filter,
            self.action_filter,
            self.diff_filter,
            max_epochs=max_epochs,
            n_samples=n_samples)

    def step(self, action):
        action = np.clip(action, self.action_bounds.lowerbound, self.action_bounds.upperbound)
        next_state = self.model.predict_state(
            self.current_state,
            action,
            self.state_filter,
            self.action_filter,
            self.diff_filter)
        reward = self.reward_func(
            self.current_state,
            next_state,
            action,
            self.name,
            is_done_func=self.model.is_done_func)
        if self.model.is_done_func:
            done = self.model.is_done_func(torch.Tensor(next_state).reshape(1,-1)).item()
        else:
            done = False
        self.current_state = next_state
        self._elapsed_steps += 1
        return next_state, reward, done, {}

    def reset(self):
        self.current_state = self.real_env.reset()
        self._elapsed_steps = 0
        return self.current_state
    
    def render(self, mode='human'):
        raise NotImplementedError


class TransitionDataset(Dataset):
    ## Dataset wrapper for sampled transitions
    def __init__(self, batch: Transition, state_filter, action_filter, diff_filter):
        state_action_filtered, _ = prepare_data(
            batch.state,
            batch.action,
            batch.nextstate,
            state_filter,
            action_filter,
            diff_filter)
        self.data_X = torch.Tensor(state_action_filtered)
        self.data_state = torch.Tensor(state_filter(batch.state))
        self.data_nextstate = torch.Tensor(state_filter(batch.nextstate))
        self.data_r = torch.Tensor(np.array(batch.reward))

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self, index):
        return self.data_X[index], self.data_state[index], self.data_nextstate[index], self.data_r[index]


class ReplayMemory(object):
    ## from torch
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        
    def push(self, transition: Transition):
        """ Saves a transition """
        self.memory.append(transition)
        
    def sample(self, batch_size: int) -> Transition:
        transitions = random.sample(self.memory, batch_size)
        return Transition(*zip(*transitions))

    def get(self, start_idx: int, end_idx: int) -> Transition:
        transitions = list(itertools.islice(self.memory, start_idx, end_idx))
        return Transition(*zip(*transitions))

    def get_all(self) -> Transition:
        return self.get(0, len(self.memory))
    
    def __len__(self) -> int:
        return len(self.memory)
    

class Ensemble(object):
    def __init__(self, params):
        
        self.params = params
        self.models = {i:Model(input_dim=params['ob_dim'] + params['ac_dim'],
                               output_dim=params['ob_dim'],
                               is_done_func = params['is_done_func'],
                               seed = params['seed'] + i,
                               num=i)
                        for i in range(params['num_models'])}
        
        self.memory = ReplayMemory(200000)
        self.memory_val = ReplayMemory(100000)
        self.train_val_ratio = 1/3
        self.is_done_func = params['is_done_func']
        weights = [weight for model in self.models.values() for weight in model.weights]
        self.optimizer = torch.optim.Adam(weights, lr=1e-3)

    def reset_models(self):
        params = self.params
        self.models = {i:Model(input_dim=params['ob_dim'] + params['ac_dim'],
                        output_dim=params['ob_dim'],
                        is_done_func = params['is_done_func'],
                        seed = params['seed'] + i,
                        num=i)
                for i in range(params['num_models'])}

    def forward(self, x: torch.Tensor):
        model_index = int(np.random.uniform()*len(self.models.keys()))
        return self.models[model_index].forward(x)
    
    def predict_state(self, state: np.array, action: np.array, state_filter, action_filter, diff_filter) -> (np.array, float):
        model_index = int(np.random.uniform()*len(self.models.keys()))
        return self.models[model_index].predict_state(state, action, state_filter, action_filter, diff_filter)
    
    def add_data(self, rollout: List[Transition]):
        for step in rollout:
            self.memory.push(step)

    def add_data_validation(self, rollout: List[Transition]):
        for step in rollout:
            self.memory_val.push(step)
    
    def check_validation_losses(self, validation_loader, diff_mean, diff_stddev):
        improved_any = False
        best_losses = []
        best_weights = []
        for model in self.models.values():
            best_losses.append(model.get_validation_loss(validation_loader, diff_mean, diff_stddev))
            best_weights.append(model.state_dict())
        best_losses = np.array(best_losses)
        improvements = best_losses < self.current_best_losses
        for i, improved in enumerate(improvements):
            if improved:
                self.current_best_losses[i] = best_losses[i]
                self.current_best_weights[i] = best_weights[i]
                improved_any = True
        return improved_any, best_losses

    def train_model(self, state_filter, action_filter, diff_filter, max_epochs: int=100, n_samples: int=200000):
        self.current_best_losses = np.zeros(self.params['num_models']) + np.inf
        self.current_best_weights = [None] * self.params['num_models']
        val_improve = deque(maxlen=6)
        if len(self.memory) < n_samples:
            n_samples = len(self.memory)
            n_samples_val = len(self.memory_val)
        else:
            n_samples_val = int(np.floor((n_samples / (1-self.train_val_ratio)) * (self.train_val_ratio)))
        samples_train = self.memory.sample(n_samples)
        samples_validate = self.memory_val.sample(n_samples_val)
        batch_size = 1024
        transition_loader = DataLoader(
            TransitionDataset(samples_train, state_filter, action_filter, diff_filter),
            shuffle=True,
            batch_size=batch_size,
            pin_memory=True
            )
        validation_loader = DataLoader(
            TransitionDataset(samples_validate, state_filter, action_filter, diff_filter),
            shuffle=False,
            batch_size=batch_size,
            pin_memory=True
            )

        diff_mean = torch.FloatTensor(diff_filter.mean).to(device)
        diff_stddev = torch.FloatTensor(diff_filter.stdev).to(device)

        ### check validation before first training epoch
        improved_any, iter_best_loss = self.check_validation_losses(validation_loader, diff_mean, diff_stddev)
        val_improve.append(improved_any)
        best_epoch = 0
        latest_best_loss = np.min(iter_best_loss)
        model_idx = 0
        print('Epoch: %s, Total Loss: N/A, Validation Loss: %s' %(0, latest_best_loss))
        for i in range(max_epochs):
            total_loss = 0
            loss = 0
            for x_batch, state_batch, nextstate_batch, r_batch in transition_loader:
                loss += self.models[model_idx].train_model_forward(x_batch, state_batch, nextstate_batch, r_batch, diff_mean, diff_stddev)
                total_loss += loss.item()
                model_idx += 1
                if model_idx >= self.params['num_models']:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    model_idx = 0
                    loss = 0
            if ((i + 1) % 5 == 0):
                improved_any, iter_best_loss = self.check_validation_losses(validation_loader, diff_mean, diff_stddev)
                latest_best_loss = np.min(iter_best_loss)
                self.valid_loss = latest_best_loss
                print('Epoch: %s, Total Loss: %s, Validation Loss: %s' %(i + 1, float(total_loss), float(latest_best_loss)))
                val_improve.append(improved_any)
                if improved_any:
                    best_epoch = (i + 1)
                if len(val_improve) > 5:
                    if not any(np.array(val_improve)[1:]):
                        assert val_improve[0]
                        print('Validation loss stopped improving at %s epochs' % (best_epoch))
                        for model_index in self.models:
                            self.models[model_index].load_state_dict(self.current_best_weights[model_index])
                        return


class Model(nn.Module):
    def __init__(self, input_dim: int, 
    					output_dim: int, 
    					h: int=1024, 
    					is_done_func=None, 
                        seed = 0,
    					num=0):

        super(Model, self).__init__()
        torch.manual_seed(seed)
        self.model = nn.Sequential(
                        nn.Linear(input_dim, h),
                        nn.ReLU(),
                        nn.Linear(h, h),
                        nn.ReLU()
                    )
        self.delta = nn.Linear(h, output_dim)
        params = list(self.model.parameters()) + list(self.delta.parameters())
        self.weights = params
        self.to(device)
        self.optimizer = torch.optim.Adam(params, lr=1e-3)
        self.loss = nn.MSELoss()
        self.memory = ReplayMemory(200000)
        self.memory_val = ReplayMemory(100000)
        self.train_val_ratio = 1/3
        self.num=str(num)
        self.is_done_func = is_done_func

    def forward(self, x: torch.Tensor):
        hidden = self.model(x)
        return self.delta(hidden)

    def predict_state(self, state: np.array, action: np.array, state_filter, action_filter, diff_filter) -> (np.array, float):
        state_filtered = state_filter(state)
        action_filtered = action_filter(action)
        if state_filtered.ndim > 1:
            major_axis = 1
        else:
            major_axis = 0
        state_action_filtered = torch.Tensor(np.concatenate((state_filtered, action_filtered), axis=major_axis)).to(device)
        delta_filtered = self.forward(state_action_filtered)
        delta_filtered = delta_filtered.cpu().detach().numpy()
        next_state_filtered = state_filtered + diff_filter.invert(delta_filtered)
        next_state = state_filter.invert(next_state_filtered)
        return next_state
            
    def train_model_forward(self, x_batch, state_batch, nextstate_batch, r_batch, diff_mean, diff_stddev):
        self.model.train()
        self.model.zero_grad()
        x_batch, state_batch, nextstate_batch, r_batch = x_batch.to(device, non_blocking=True), state_batch.to(device, non_blocking=True), nextstate_batch.to(device, non_blocking=True), r_batch.to(device, non_blocking=True)
        y_pred = self.forward(x_batch)
        nextstate_pred = (y_pred * diff_stddev) + diff_mean + state_batch
        loss = self.loss(nextstate_pred, nextstate_batch)
        return loss

    def get_validation_loss(self, validation_loader, diff_mean, diff_stddev):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x_batch_val, state_batch_val, nextstate_batch_val, _ in validation_loader:
                x_batch_val, state_batch_val, nextstate_batch_val = x_batch_val.to(device, non_blocking=True), state_batch_val.to(device, non_blocking=True), nextstate_batch_val.to(device, non_blocking=True)
                y_pred_val = self.forward(x_batch_val)
                nextstate_pred_val = (y_pred_val * diff_stddev) + diff_mean + state_batch_val
                total_loss += self.loss(nextstate_pred_val, nextstate_batch_val)
        return total_loss.item()

    def get_acquisition(self, rollouts: List[List[Transition]], state_filter, action_filter, diff_filter):
        self.model.eval()
        state = []
        action = []
        nextstate = []
        for rollout in rollouts:
            for step in rollout:
                state.append(step.state)
                action.append(step.action)
                nextstate.append(step.nextstate)
        state = np.array(state)
        state_filtered = state_filter(state)
        action = np.array(action)
        nextstate = np.array(nextstate)
        nextstate_filtered = state_filter(nextstate)
        state_action_filtered, _ = prepare_data(
            state,
            action,
            nextstate,
            state_filter,
            action_filter,
            diff_filter)
        state_action_filtered = torch.Tensor(state_action_filtered).to(device)
        delta_pred = self.forward(state_action_filtered)
        delta_pred = delta_pred.cpu().detach().numpy()
        nextstate_pred = state_filtered + diff_filter.invert(delta_pred)
        return float(self.loss(torch.Tensor(nextstate_pred), torch.Tensor(nextstate_filtered)).item())
