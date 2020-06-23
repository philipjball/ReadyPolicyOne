import uuid
from collections import deque

import numpy as np
import pandas as pd
import torch

from model import Transition
from ppo import PPO
from utils import (filter_torch, filter_torch_invert, get_residual, get_stats,
                   random_env_forward, torch_reward)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def collect_data(params, ppo, memory, ensemble_env):
    rollouts = []
    timesteps = 0
    env = ensemble_env.real_env
    collection_timesteps = params['outer_steps']
    pca_data = []
    # Standard RL interaction loop with the real env
    residual = 1
    while timesteps < collection_timesteps:
        if not params['fix_std']:
            new_std = np.random.uniform(0.0, 3.0)
            ppo.change_policy_std(new_std)
        rollout = []
        done = False
        env_ts = 0
        state = env.reset()
        ensemble_env.state_filter.update(state)
        newdata = []
        while (not done):
            action = ppo.select_action(ensemble_env.state_filter(state), memory)
            newdata.append(np.concatenate((state, action)))
            nextstate, reward, done, _ = env.step(action)
            rollout.append(Transition(state, action, reward, nextstate))
            state = nextstate
            ensemble_env.state_filter.update(state)
            ensemble_env.action_filter.update(action)
            timesteps += 1
            env_ts += 1
            
            if residual < params['pca']:
                collection_timesteps = 0

            if (timesteps) % 100 == 0:
                print("Collected Timesteps: %s" %(timesteps))                
        
        if len(pca_data) > 0:
            residual, train_resid = get_residual(newdata, pca_data, 0.99)
            print("Residual = {}, Train Residual = {}".format(str(residual), str(train_resid)))
        pca_data += newdata
        rollouts.append(rollout)
    
    num_valid = int(np.floor(ensemble_env.model.train_val_ratio * len(rollouts)))
    train = rollouts[(num_valid):]
    valid = rollouts[:num_valid]
    for rollout in train:
        ensemble_env.model.add_data(rollout)
    for rollout in valid:
        ensemble_env.model.add_data_validation(rollout)
    print("\nAdded {} samples to the model, {} for valid".format(str(len(train)), str(len(valid))))
        
    ensemble_env.update_diff_filter()
    errors = [ensemble_env.model.models[i].get_acquisition(rollouts, ensemble_env.state_filter, ensemble_env.action_filter, ensemble_env.diff_filter) for i in range(params['num_models'])]
    error =  np.sqrt(np.mean(np.array(errors)**2))
    print("\nMSE Loss on new rollouts: %s" % error)
    return(timesteps, error)


def train_agent(ppo: PPO, env, policy_iters, max_timesteps, memory, update_timestep, env_resets, log_interval, lam=0, n_parallel=500, var_type='reward'):
    running_reward = 0
    avg_length = 0
    time_step = 0
    n_updates = 0
    i_episode = 0
    prev_performance = np.array([-np.inf for _ in range(len(env.model.models))])
    memory.clear_memory()
    rewards_history = deque(maxlen=6)
    best_weights = None
    is_done_func = env.model.is_done_func
    if var_type == 'reward':
        state_dynamics = False
    elif var_type == 'state':
        state_dynamics = True
    else:
        raise Exception("Variance must either be 'reward' or 'state'")

    for model in env.model.models.values():
        model.to(device)

    state_mean = torch.FloatTensor(env.state_filter.mean).to(device)
    state_stddev = torch.FloatTensor(env.state_filter.stdev).to(device)
    action_mean = torch.FloatTensor(env.action_filter.mean).to(device)
    action_stddev = torch.FloatTensor(env.action_filter.stdev).to(device)
    diff_mean = torch.FloatTensor(env.diff_filter.mean).to(device)
    diff_stddev = torch.FloatTensor(env.diff_filter.stdev).to(device)

    start_states = torch.FloatTensor(env_resets).to(device)

    done_true = [True for _ in range(n_parallel)]
    done_false = [False for _ in range(n_parallel)]
    
    while n_updates < policy_iters:
        i_episode += n_parallel
        state = start_states.clone()
        prev_done = done_false
        var = 0
        t = 0
        while t < max_timesteps:
            state_f = filter_torch(state, state_mean, state_stddev)
            time_step += n_parallel
            t += 1
            with torch.no_grad():
                action = ppo.policy_old.act(state_f, memory)
                action = torch.clamp(action, env.action_bounds.lowerbound[0], env.action_bounds.upperbound[0])
                action_f = filter_torch(action, action_mean, action_stddev)
                X = torch.cat((state_f, action_f), dim=1)
                y = random_env_forward(X, env)
            nextstate_f = state_f + filter_torch_invert(y, diff_mean, diff_stddev)
            nextstate = filter_torch_invert(nextstate_f, state_mean, state_stddev)
            if is_done_func:
                done = is_done_func(nextstate).cpu().numpy()
                done[prev_done] = True
                prev_done = done
            else:
                if t >= max_timesteps:
                    done = done_true
                else:
                    done = done_false
            uncert = get_stats(env, X, state_f, action, diff_mean, diff_stddev, state_mean, state_stddev, done, state_dynamics)
            reward = torch_reward(env.name, nextstate, action, done)
            reward = (1-lam) * reward + lam * uncert
            state = nextstate
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            running_reward += reward
            var += uncert**2
            # update if it's time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
                n_updates += 1
                if n_updates > 10:
                    improved, prev_performance = validate_agent_with_ensemble(ppo, env, start_states, state_mean, state_stddev, action_mean, action_stddev, diff_mean, diff_stddev, prev_performance, 0.7, memory, max_timesteps)
                    if improved:
                        best_weights = ppo.policy.state_dict()
                        best_update = n_updates
                    rewards_history.append(improved)
                    if len(rewards_history) > 5:
                        if rewards_history[0] > max(np.array(rewards_history)[1:]):
                            print('Policy Stopped Improving after {} updates'.format(best_update))
                            ppo.policy.load_state_dict(best_weights)
                            ppo.policy_old.load_state_dict(best_weights)
                            return
        avg_length += t * n_parallel
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward.sum()/log_interval))           
            print('Episode {} \t Avg length: {} \t Avg reward: {} \t Number of Policy Updates: {}'.format(i_episode, avg_length, running_reward, n_updates))
            running_reward = 0
            avg_length = 0


def validate_agent_with_ensemble(ppo, env, start_states, state_mean, state_stddev, action_mean, action_stddev, diff_mean, diff_stddev, best_performance, threshold, memory, ep_steps):

    n_parallel = start_states.shape[0]

    performance = np.zeros(len(env.model.models))
    is_done_func = env.model.is_done_func

    done_true = [False for _ in range(n_parallel)]
    done_false = [False for _ in range(n_parallel)]

    for i in env.model.models:
        total_reward = 0
        state = start_states.clone()
        prev_done = done_false
        t = 0
        while t < ep_steps:
            state_f = filter_torch(state, state_mean, state_stddev)
            t += 1
            with torch.no_grad():
                action = ppo.policy_old.act(state_f, memory, False)
                action = torch.clamp(action, env.action_bounds.lowerbound[0], env.action_bounds.upperbound[0])
                action_f = filter_torch(action, action_mean, action_stddev)
                X = torch.cat((state_f, action_f), dim=1)
                y = env.model.models[i].forward(X)
            nextstate_f = state_f + filter_torch_invert(y, diff_mean, diff_stddev)
            nextstate = filter_torch_invert(nextstate_f, state_mean, state_stddev)
            if is_done_func:
                done = is_done_func(nextstate).cpu().numpy()
                done[prev_done] = True
                prev_done = done
            else:
                if t >= ep_steps:
                    done = done_true
                else:
                    done = done_false   
            reward = torch_reward(env.name, nextstate, action, done)        
            state = nextstate
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            total_reward += np.mean(reward)
        performance[i] = total_reward
    memory.clear_memory()
    if (np.mean(performance > best_performance) > threshold):
        new_best_performance = np.maximum(performance, best_performance)
        return True, new_best_performance
    else:
        new_best_performance = best_performance
        return False, new_best_performance


def test_agent(ppo, env, memory, ep_steps, subset_resets, subset_real_resets, use_model):
    num_rollouts = len(subset_resets)
    if use_model:
        test_env = env
    else:
        test_env = env.real_env
        half = int(np.ceil(len(subset_real_resets[0]) / 2))
    total_reward = 0
    for reset, real_reset in zip(subset_resets, subset_real_resets):
        time_step = 0
        done = False
        test_env.reset()
        state = reset
        if use_model:
            test_env.current_state = state
        else:
            test_env.env.unwrapped.set_state(real_reset[:half], real_reset[half:])
        while (not done) and (time_step < ep_steps):
            time_step += 1
            action = ppo.select_action(env.state_filter(state), memory, False)
            state, reward, done, _ = test_env.step(action)
            memory.rewards.append(reward)
            memory.is_terminals.append(False)
            total_reward += reward
    memory.clear_memory()            
    return total_reward / num_rollouts


def train_agent_model_free(ppo, ensemble_env, memory, update_timestep, seed, log_interval, ep_steps, start_states, start_real_states):
    # logging variables
    running_reward = 0
    running_reward_real = 0
    avg_length = 0
    time_step = 0
    cumulative_update_timestep = 0
    cumulative_log_timestep = 0
    n_updates = 0
    i_episode = 0
    log_episode = 0
    samples_number = 0
    samples = []
    rewards = []
    n_starts = len(start_states)

    env_name = ensemble_env.unwrapped.spec.id

    state_filter = ensemble_env.state_filter

    half = int(np.ceil(len(start_real_states[0]) / 2))

    env = ensemble_env.real_env

    memory.clear_memory()

    while samples_number < 3e7:
        for reset, real_reset in zip(start_states, start_real_states):
            time_step = 0
            done = False
            env.reset()
            state = reset
            env.unwrapped.set_state(real_reset[:half], real_reset[half:])
            i_episode += 1
            log_episode += 1
            state = env.reset()
            state_filter.update(state)
            state = state_filter(state)
            done = False

            while (not done):
                cumulative_log_timestep += 1
                cumulative_update_timestep += 1
                time_step += 1
                samples_number += 1
                action = ppo.select_action(state_filter(state), memory)
                nextstate, reward, done, _ = env.step(action)
                state = nextstate
                state_filter.update(state)

                memory.rewards.append(np.array([reward]))
                memory.is_terminals.append(np.array([done]))

                running_reward += reward

                # update if it's time
                if cumulative_update_timestep % update_timestep == 0:
                    ppo.update(memory)
                    memory.clear_memory()
                    cumulative_update_timestep = 0
                    n_updates += 1

            # logging
            if i_episode % log_interval == 0:
                subset_resets_idx = np.random.randint(0, n_starts, 10)
                subset_resets = start_states[subset_resets_idx]
                subset_resets_real = start_real_states[subset_resets_idx]
                avg_length = int(cumulative_log_timestep/log_episode)
                running_reward = int((running_reward_real/log_episode))
                actual_reward = test_agent(ppo, ensemble_env, memory, ep_steps, subset_resets, subset_resets_real, use_model=False)
                samples.append(samples_number)
                rewards.append(actual_reward)
                print('Episode {} \t Samples {} \t Avg length: {} \t Avg reward: {} \t Actual reward: {} \t Number of Policy Updates: {}'.format(i_episode, samples_number, avg_length, running_reward, actual_reward, n_updates))
                df = pd.DataFrame({'Samples': samples, 'Reward': rewards})
                df.to_csv("{}.csv".format(env_name + '-ModelFree-Seed-' + str(seed)))
                cumulative_log_timestep = 0
                log_episode = 0
                running_reward = 0


def train_agent_model_free_debug(ppo, ensemble_env, memory, update_timestep, log_interval, reward_func=None):
    # logging variables
    running_reward = 0
    running_reward_no_filter = 0
    running_reward_real = 0
    avg_length = 0
    time_step = 0
    cumulative_update_timestep = 0
    cumulative_log_timestep = 0
    n_updates = 0
    i_episode = 0
    log_episode = 0
    samples_number = 0
    samples = []
    rewards = []
    rewards_real = []

    seed = 0
    env_name = ensemble_env.unwrapped.spec.id

    state_filter = ensemble_env.state_filter

    env = ensemble_env.real_env

    if hasattr(env, 'is_done_func'):
        is_done_func = env.is_done_func
    else:
        is_done_func = None

    memory.clear_memory()

    while samples_number < 2e7:
        i_episode += 1
        log_episode += 1
        state = env.reset()
        state_filter.update(state)
        state = state_filter(state)
        done = False
        while (not done):
            cumulative_log_timestep += 1
            cumulative_update_timestep += 1
            time_step += 1
            samples_number += 1
            action = ppo.select_action(state_filter(state), memory)
            nextstate, reward, done, _ = env.step(action)
            running_reward_no_filter += reward_func(state, nextstate, action, env_name, is_done_func=is_done_func)
            running_reward_real += reward
            reward = reward_func(state_filter(state), state_filter(nextstate), action, env_name, state_filter, is_done_func=is_done_func)
            state = nextstate
            state_filter.update(state)

            memory.rewards.append(np.array([reward]))
            memory.is_terminals.append(np.array([done]))

            running_reward += reward

            # update if it's time
            if cumulative_update_timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                cumulative_update_timestep = 0
                n_updates += 1

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(cumulative_log_timestep/log_episode)
            running_reward = int((running_reward/log_episode))
            running_reward_no_filter = int((running_reward_no_filter/log_episode))
            running_reward_real = int((running_reward_real/log_episode))
            samples.append(samples_number)
            rewards.append(running_reward)
            rewards_real.append(running_reward_real)
            print('Episode {} \t Samples {} \t Avg length: {} \t Avg reward: {} \t Avg reward no filter: {} \t Avg real reward: {} \t Number of Policy Updates: {}'.format(i_episode, samples_number, avg_length, running_reward, running_reward_no_filter, running_reward_real, n_updates))
            df = pd.DataFrame({'Samples': samples, 'Reward': rewards, 'Reward_Real': rewards_real})
            df.to_csv("{}.csv".format(env_name + '-ModelFree-Seed-' + str(seed)))
            cumulative_log_timestep = 0
            log_episode = 0
            running_reward = 0
            running_reward_no_filter = 0
            running_reward_real = 0

        time_step = 0
