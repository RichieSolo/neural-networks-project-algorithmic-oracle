#import sys
#sys.path.append('C:\\Users\\Clint\\neural-networks-project-algorithmic-oracle')
from utils.data import *

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env

class GoLeftEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    Based on perecent return matrix method from 4/20 zoom call.
    """

    metadata = {"render_modes": ["console"]} # Not doing anything with this yet...

    stock_df = [];
    initial_wealth = 1;
    wealth = [];
    window_size = 60; # days
    step_size = 5; # days
    grid_size = 21;
    table_ix = 0;

    def __init__(self, stock_df=[], window_size = 60, step_size = 5, grid_size=21, initial_wealth=1, render_mode="console"):
        super(GoLeftEnv, self).__init__()
        self.render_mode = render_mode

        self.stock_df = stock_df;
        self.window_size = window_size;
        self.step_size = step_size;
        self.initial_wealth = initial_wealth;
        self.wealth = np.array([initial_wealth]);
        self.grid_size = grid_size

        # Initialize the agent
        self.agent_pos = initial_wealth

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        self.n_stocks = stock_df.shape[1]
        self.action_space = spaces.Box(
            low=0, high=self.grid_size, shape=(1,self.n_stocks), dtype=np.float32
        )

        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(window_size,self.n_stocks+1), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        # Initialize the agent at the right of the grid
        return_mat = self.stock_df.to_numpy()[:self.window_size,:];
        self.agent_pos = np.c_[np.zeros(self.window_size), return_mat]
        self.table_ix = self.window_size+1;

        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array(self.agent_pos).astype(np.float32), {}  # empty info dict

    def step(self, action):

        n = self.step_size

        w = np.c_[np.floor(self.grid_size/2), action]
        w = w/w.sum();

        X = self.stock_df.to_numpy()[self.window_size:self.window_size+n,:];
        R = np.dot(X, w[0,1:].T)

        self.wealth = np.r_[self.wealth, self.wealth[-1] * np.cumprod(1 + R).squeeze()];
        percent_returns = np.diff(self.wealth[-(n+1):]) / self.wealth[-n:];

        self.agent_pos = np.r_[self.agent_pos[n:,:], np.c_[percent_returns, X]];
        # Are we at the left of the grid?
        terminated = bool(self.wealth[-1] <= 0)
        truncated = False  # we do not limit the number of steps here

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = percent_returns.prod();

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (
            np.array(self.agent_pos).astype(np.float32),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "console":
            print(self.wealth[-1:])

    def close(self):
        pass


data = get_sp500_df()
print(data)

stocks_df = data['stocks']
stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])

# key parsing pivot step
pivot_stocks_df = stocks_df.pivot(index='Date', columns='Symbol', values='Close')

stocks = ['AAPL', 'BA'];

daily_returns = pivot_stocks_df.loc[:,stocks].pct_change().dropna()

env = GoLeftEnv(stock_df=daily_returns, grid_size=21)

check_env(env, warn=True)

obs, _ = env.reset()
env.render()

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())

# Hardcoded agent: always invest in stock 1
n_steps = 20
for step in range(n_steps):
    print(f"Step {step + 1}")
    obs, reward, terminated, truncated, info = env.step(np.array([[0, 1]]))
    done = terminated or truncated
    #print("obs=", obs, "reward=", reward, "done=", done)
    env.render()
    if done:
        print("Goal reached!", "reward=", reward)
        break


# Instantiate the env
vec_env = make_vec_env(GoLeftEnv, n_envs=1, env_kwargs=dict(stock_df=daily_returns, grid_size=21))

# Train the agent
model = A2C("MlpPolicy", env, verbose=1).learn(5000)

# Test the trained agent
# using the vecenv
obs = vec_env.reset()
n_steps = 20
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print(f"Step {step + 1}")
    print("Action: ", action)
    obs, reward, done, info = vec_env.step(action)
    #print("obs=", obs, "reward=", reward, "done=", done)
    vec_env.render()
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break