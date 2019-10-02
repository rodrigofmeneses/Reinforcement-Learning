#%%
import gym
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import deque
#%%
class MountainCar():
    def __init__(self, buckets=(20 , 15,), n_episodes=1000, n_win_ticks=195, min_alpha=0.1, min_epsilon=0.1, gamma=1.0, ada_divisor=25, max_env_steps=None, monitor=False, render=False):
        self.buckets = buckets # down-scaling feature space to discrete range
        self.n_episodes = n_episodes # training episodes 
        self.n_win_ticks = n_win_ticks # average ticks over 100 episodes required for win
        self.min_alpha = min_alpha # learning rate
        self.min_epsilon = min_epsilon # exploration rate
        self.gamma = gamma # discount factor
        self.ada_divisor = ada_divisor # only for development purposes
        self.render = render
        self.env = gym.make('MountainCar-v0')
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        if monitor: self.env = gym.wrappers.Monitor(self.env, 'tmp/cartpole-1', force=True) # record results for upload

        # initialising Q-table
        self.Q = np.zeros(self.buckets + (self.env.action_space.n,))
        self.rewards = []

    # Discretizing input space to make Q-table and to reduce dimmensionality
    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1]]
        lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1]]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    # Choosing action based on epsilon-greedy policy
    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[state])

    # Updating Q-value of state-action pair based on the update equation
    def update_q(self, state_old, action, reward, state_new, alpha):
        self.Q[state_old][action] += alpha * (reward + self.gamma * np.max(self.Q[state_new]) - self.Q[state_old][action])

    # Adaptive learning of Exploration Rate
    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    # Adaptive learning of Learning Rate
    def get_alpha(self, t):
        return max(self.min_alpha, min(1.0, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    def get_rewards(self):
        return self.rewards
    
    def run(self):

        for e in range(self.n_episodes):
            # As states are continuous, discretize them into buckets
            current_state = self.discretize(self.env.reset())

            # Get adaptive learning alpha and epsilon decayed over time
            alpha = self.get_alpha(e)
            epsilon = self.get_epsilon(e)
            done = False
            cumulative_reward = 0
            i = 0

            while not done:
                # Render environment
                if self.render: self.env.render()

                # Choose action according to greedy policy and take it
                action = self.choose_action(current_state, epsilon)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize(obs)

                # Update Q-Table
                self.update_q(current_state, action, reward, new_state, alpha)
                current_state = new_state
                cumulative_reward += reward
                i += 1
            # print(f"Episode {e}: cumulative reward {cumulative_reward}")
            self.rewards.append(cumulative_reward) 

    def bestRun(self):
        current_state = self.discretize(self.env.reset())
        done = False

        while not done:
            # Render environment
            self.env.render()

            # Choose action according to greedy policy and take it
            action = np.argmax(self.Q[current_state])
            obs, reward, done, _ = self.env.step(action)
            current_state = self.discretize(obs)

    def close(self):
        self.env.close()

#%%

if __name__ == "__main__":

    # Make an instance of MountainCar class 
    buckets=(16, 6,) # Great Bucket
    solver = MountainCar(buckets=buckets, n_episodes=3000, render=False)
    solver.run()
    
    plt.plot(solver.get_rewards())
    plt.title("Taxa de aprendizado")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.show()

    solver.bestRun()
    solver.close()


#%%
