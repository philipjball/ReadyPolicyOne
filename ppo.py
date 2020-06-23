import torch
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPODataSet(Dataset):
    def __init__(self, states, actions, logprobs, rewards):
        self.states = states
        self.actions = actions
        self.logprobs = logprobs
        self.rewards = rewards

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index], self.actions[index], self.logprobs[index], self.rewards[index]
    

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, seed, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.action_dim = action_dim

        torch.manual_seed(seed)
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim),
                nn.Tanh()
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
                )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory, stochastic=True):
        action_mean = self.actor(state)

        cov_mat = torch.diag(self.action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        if not stochastic:
            action = action_mean
            
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, state, action):   
        action_mean = torch.squeeze(self.actor(state))
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, seed, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.seed = seed
        self.policy = ActorCritic(self.seed, self.state_dim, self.action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.policy_old = ActorCritic(self.seed, self.state_dim, self.action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def change_policy_std(self, new_std):
        action_dim = self.policy_old.action_dim
        self.policy_old.action_var = torch.full((action_dim,), new_std * new_std).to(device)
    
    def select_action(self, state, memory, stochastic=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory, stochastic).cpu().data.numpy().flatten()
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        memory_rewards_unrolled = np.concatenate([col for col in np.stack(memory.rewards).T])
        memory_is_terminals_unrolled = np.concatenate([col for col in np.stack(memory.is_terminals).T])
        # need to remove entries of episodes that terminated before the time limit
        flip_idx = (memory_is_terminals_unrolled[:-1] != memory_is_terminals_unrolled[1:])
        flip_idx = np.insert(flip_idx, 0, False)
        keep_idx = ~memory_is_terminals_unrolled
        keep_idx[flip_idx] = True
        memory_rewards_unrolled = memory_rewards_unrolled[keep_idx]
        memory_is_terminals_unrolled = memory_is_terminals_unrolled[keep_idx]

        # discounted_reward = torch.zeros(memory.rewards[0].shape).to(device)
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory_rewards_unrolled), reversed(memory_is_terminals_unrolled)):
            if is_terminal:
                discounted_reward = 0
            # discounted_reward[is_terminal] = 0 FOR VECTORISED FORM WHICH WE'LL DO LATER
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        keep_idx = torch.tensor(keep_idx)
        memory_states_unrolled = torch.cat([col for col in torch.stack(memory.states).permute(1,0,2)])[keep_idx]
        memory_actions_unrolled = torch.cat([col for col in torch.stack(memory.actions).permute(1,0,2)])[keep_idx]
        memory_logprobs_unrolled = torch.cat([col for col in torch.stack(memory.logprobs).T])[keep_idx]

        # convert list to tensor
        old_states_full = torch.squeeze(memory_states_unrolled.to(device).reshape(-1,self.state_dim)).detach()
        old_actions_full = torch.squeeze(memory_actions_unrolled.to(device).reshape(-1,self.action_dim)).detach()
        old_logprobs_full = torch.squeeze(memory_logprobs_unrolled.to(device).reshape(-1,1)).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_full, old_actions_full)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs_full.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
