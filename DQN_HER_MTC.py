from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.envs import BitFlippingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from gym.envs.registration import EnvSpec
from sb3_contrib.qrdqn import QRDQN
import gym
from gym import Env, spaces
import numpy as np

model_class = DQN  # works also with DQN ,SAC, DDPG and TD3, QRDQN
MTC = gym.make("MountainCar-v0")
MTC._max_episode_steps = 1000

class MTC_for_HER(Env):
    '''
    he position is clipped to the range `[-1.2, 0.6]`
    velocity is clipped to the range `[-0.07, 0.07]`
    '''
    spec = EnvSpec("MountainCartWithGoal-v0")
    def __init__(self, mtc) -> None:
        super().__init__()
        self.MTC = mtc
        self.desired_goal = [0.6, 0.05]
        self.observation_space = spaces.Dict(
            {
                "observation": self.MTC.observation_space,
                "achieved_goal": self.MTC.observation_space,
                "desired_goal": self.MTC.observation_space,
            }
        )
        self.action_space = self.MTC.action_space
        self.MaxObsBuffer = 100000
        self.ObsBuffer = []
    def step(self, action):
        obs, reward, done, info = self.MTC.step(action)
        # First adjust obs
        self.new_obs = {
            "observation": obs, 
            "achieved_goal": obs, 
            "desired_goal" :self.desired_goal
        }
        obs = self.new_obs
        # second adjust reward by checking whether desired result have reached or not
        reward += self.compute_reward(self.new_obs['achieved_goal'], self.new_obs['desired_goal'], None)
        # print('obs', obs['observation'])
        
        # randi  = np.random.rand()
        # if randi < 0.01:
        #     self.ObsBuffer.append(obs['observation'])
        # if len(self.ObsBuffer) > self.MaxObsBuffer:
        #     del self.ObsBuffer[0]
        # print('self.ObsBuffer', len(self.ObsBuffer))
        return obs, reward, done, info
    def GetObs(self, obs):
        new_obs = {
            "observation": obs, 
            "achieved_goal": obs, 
            "desired_goal" :self.desired_goal
        }
        

        return new_obs


    def compute_reward(self, achieved_goal, desired_goal, info):
        # print('achieved_goal, desired_goal',achieved_goal, desired_goal)
        # print('achieved_goal,', type(achieved_goal))
        # print('achieved_goal', len(achieved_goal.shape))
        '''
        HER  reward
        Once achive  then all achive
        '''
        
        if len(achieved_goal.shape) == 1:
            if achieved_goal[0] > desired_goal[0]:
                return 1
            else:
                return 0
        if len(achieved_goal.shape) == 2:
            # print('desired_goal, ', desired_goal)
            # print('desired_goal, ', desired_goal.shape)
            return (achieved_goal[:, 0] - desired_goal[:,0] > 0) * 1
        
    def reset(self):
        obs = self.MTC.reset()
        return self.GetObs(obs)
    def render(self, mode: str = "human") :
        return self.MTC.state.copy()

    def close(self) -> None:
        pass


class MTC_modified_Reward(Env):
    '''
    he position is clipped to the range `[-1.2, 0.6]`
    velocity is clipped to the range `[-0.07, 0.07]`
    '''
    spec = EnvSpec("MountainCartWithGoal-v0")
    def __init__(self, mtc) -> None:
        super().__init__()
        self.MTC = mtc
        self.desired_goal = [0.6, 0.05]
        self.observation_space = spaces.Dict(
            {
                "observation": self.MTC.observation_space,
                "achieved_goal": self.MTC.observation_space,
                "desired_goal": self.MTC.observation_space,
            }
        )
        self.action_space = self.MTC.action_space
        self.highest_pos = -10000
    def step(self, action):
        obs, reward, done, info = self.MTC.step(action)
        # First adjust obs
        self.new_obs = {
            "observation": obs, 
            "achieved_goal": obs, 
            "desired_goal" :self.desired_goal
        }
        obs = self.new_obs
        # second adjust reward by checking whether desired result have reached or not
        reward += self.compute_reward(self.new_obs['achieved_goal'], self.new_obs['desired_goal'], None)
        # if obs['observation'][0] > self.highest_pos:
        #     self.highest_pos = obs['observation'][0]
        return obs, reward, done, info
    def GetObs(self, obs):
        new_obs = {
            "observation": obs, 
            "achieved_goal": obs, 
            "desired_goal" :self.desired_goal
        }
        

        return new_obs


    def compute_reward(self, achieved_goal, desired_goal, info):
        self.count += 1
        # print('achieved_goal, desired_goal',achieved_goal, desired_goal)
        # print('achieved_goal,', type(achieved_goal))
        # print('achieved_goal', len(achieved_goal.shape))
        
        if len(achieved_goal.shape) == 1:
            if achieved_goal[0] > self.highest_pos:
                return self.highest_pos - self.count
            else:
                return 0

        if len(achieved_goal.shape) == 2:
            # print('achieved_goal', achieved_goal.shape)
            # print('self.highest_pos', self.highest_pos)
            if self.highest_pos.shape != desired_goal[:,0].shape:
                self.highest_pos = desired_goal[:,0]* 0 - 1000
            return (desired_goal[:,0] > self.highest_pos) * 1 * self.highest_pos - self.count
        
    def reset(self):
        self.count = 0
        obs = self.MTC.reset()
        # print('obs', obs)
        return self.GetObs(obs)
    def render(self, mode: str = "human") :
        return self.MTC.state.copy()

    def close(self) -> None:
        pass

# env = MTC_modified_Reward(MTC)
env = MTC_for_HER(MTC)


# Available strategies (cf paper): future, final, episode, uniform
goal_selection_strategy = "final" # equivalent to GoalSelectionStrategy.FUTURE, uniform

# If True the HER transitions will get sampled online
online_sampling = True
# Time limit for the episodes
max_episode_length = 1000

# Initialize the model
model = model_class(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    batch_size = 50,
    learning_starts = 5000,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
        online_sampling=online_sampling,
        max_episode_length=max_episode_length,
    ),
    verbose=1,
    buffer_size = 1000000
)

# Train the model
model.learn(500000)

model.save("./her_bit_env")
# Because it needs access to `env.compute_reward()`
# HER must be loaded with the envz
model = model_class.load("./her_bit_env", env=env)


def runall():
    obs = env.reset()

    mtc_obs = MTC.reset()
    # print('MTC.env.state', MTC.env.state)
    # print('obs', obs)
    # breakpoint()
    MTC.env.state =  obs['observation']#.tolist()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, _ = env.step(action)
        # print('obs', obs)
        _, reward, done, _ = MTC.step(action)
        MTC.render()
        if done:
            obs = env.reset()
runall()
breakpoint()