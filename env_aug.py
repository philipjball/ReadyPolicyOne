from gym.envs.mujoco import HalfCheetahEnv, AntEnv, HopperEnv
import numpy as np
from gym import utils
import os
import torch
from gym.envs.mujoco import mujoco_env
from gym.envs.registration import EnvSpec


class HalfCheetahEnvAug(HalfCheetahEnv):

    def __init__(self):
        self.aug_vel = 0
        super(HalfCheetahEnvAug, self).__init__()
        self.spec = EnvSpec('HalfCheetah-v2')
    
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            np.array([self.aug_vel])
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.aug_vel = 0
        return self._get_obs()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        self.aug_vel = (xposafter - xposbefore)/self.dt
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

class AntEnvAug(AntEnv):

    def __init__(self):
        super(AntEnvAug, self).__init__()
        self.aug_vel = 0
        self.spec = EnvSpec('Ant-v2')
    
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.array([self.aug_vel])
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.aug_vel = 0
        return self._get_obs()

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        self.aug_vel = (xposafter - xposbefore)/self.dt
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        # In the true env we keep the contact cost so we can compare to other algorithms
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)
    
    @staticmethod
    def is_done_func(states):
        finite_check = torch.isfinite(states).all(dim=1)
        bounds_check = (states[:,0] >= 0.2) & (states[:,0] <= 1.0)
        notdone = finite_check & bounds_check
        return ~notdone


class HopperEnvAug(HopperEnv):

    def __init__(self):
        self.aug_vel = 0
        super(HopperEnvAug, self).__init__()
        self.spec = EnvSpec('Hopper-v2')
    
    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        self.aug_vel = reward
        reward += alive_bonus
        reward -= 1e-1 * np.square(a).sum()
        height_penalty = -3.0 * (height - 1.3)**2
        reward += height_penalty
        s = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10),
            np.array([self.aug_vel])
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.aug_vel = 0
        return self._get_obs()


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class fixedSwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/fixed_swimmer.xml' % dir_path, 4)
        utils.EzPickle.__init__(self)
        self.spec = EnvSpec('Swimmer-v2')
        self.pos_diff = 0

    def step(self, a):
        ctrl_cost_coeff = 0.0001

        """
        xposbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.model.data.qpos[0, 0]
        """

        self.xposbefore = self.sim.data.site_xpos[0][0] / self.dt
        self.do_simulation(a, self.frame_skip)
        self.xposafter = self.sim.data.site_xpos[0][0] / self.dt
        self.pos_diff = self.xposafter - self.xposbefore

        reward_fwd = self.xposafter - self.xposbefore
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat, np.array([self.pos_diff])])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        self.pos_diff = 0
        return self._get_obs()