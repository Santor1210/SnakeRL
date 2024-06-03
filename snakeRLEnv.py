import gym
from gym import spaces
import numpy as np
from snakeRL import Pygame2D
import time
from collections import deque


# Agent
# Reward
# Environment
# Action

class CustomEnv(gym.Env):
    def __init__(self, mode='bot'):
        """Set the action and observation spaces Initialise the pygame object"""
        super().__init__()
        self.mode = mode
        self.grid_size = 20
        self.pygame = Pygame2D(self.grid_size, mode=self.mode)
        self.action_space = spaces.Discrete(4)  # 0-->right, 1-->left, 2-->up, 3-->down
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.grid_size * 2 - 1, self.grid_size * 2 - 1, 1),
                                            dtype=np.uint8)
        self.episode_reward = 0
        self.num_episodes = 0
        self.last_hundred_ep_rewards = deque()
        self.last_thousand_ep_rewards = deque()
        self.last_ten_thousand_ep_rewards = deque()
        self.steps_counter = 0
        self.start_time = time.time()
        self.buffer_Full = True

    def get_action(self):
        """helper function to return keyboard inputs"""
        return self.pygame.get_human_action()

    def step(self, action):
        """Take action & update the env Return -> next-state, reward, done, info"""
        self.pygame.action(action)
        obs = self.pygame.observe()
        reward = self.pygame.evaluate()
        self.episode_reward += reward
        self.steps_counter += 1
        done = self.pygame.is_done()
        return obs, float(reward), done, {}

    def render(self, mode='bot', close=False):
        """Render the env on the screen"""
        self.pygame.view()

    def reset(self):
        """Re-initialise the pygame object Return -> starting state of env"""
        if self.buffer_Full:
            self.last_hundred_ep_rewards.append(self.episode_reward)
            self.last_thousand_ep_rewards.append(self.episode_reward)
            self.last_ten_thousand_ep_rewards.append(self.episode_reward)

            if len(self.last_hundred_ep_rewards) > 25:
                self.last_hundred_ep_rewards.popleft()
            if len(self.last_thousand_ep_rewards) > 1000:
                self.last_thousand_ep_rewards.popleft()
            if len(self.last_ten_thousand_ep_rewards) > 10000:
                self.last_ten_thousand_ep_rewards.popleft()

            self.episode_reward = 0

            if self.num_episodes % 25 == 0 and self.num_episodes > 0:
                hundred_m_reward = format(np.mean(self.last_hundred_ep_rewards), '.2f')
                thousand_m_reward = format(np.mean(self.last_thousand_ep_rewards), '.2f')
                ten_thousand_m_reward = format(np.mean(self.last_ten_thousand_ep_rewards), '.2f')
                time_elapsed = int(time.time() - self.start_time)
                fps = int(self.steps_counter / time_elapsed)
                steps_avg = int(self.steps_counter / self.num_episodes)
           #     print(f'Ep {self.num_episodes}', f'---Step {self.steps_counter}', f'---Steps_Avg {steps_avg}',
           #           f'---FPS {fps}', f'---M_Reward {ten_thousand_m_reward}', f'---M_25_Reward {hundred_m_reward}',
           #           f'---M_1K_Reward {thousand_m_reward}', f'---Time {time_elapsed}')
            self.num_episodes += 1
        elif self.steps_counter >= 100000 and not self.buffer_Full:
            self.buffer_Full = True
            self.episode_reward = 0
            self.steps_counter = 0
            self.start_time = time.time()

        del self.pygame
        self.pygame = Pygame2D(self.grid_size, mode=self.mode)
        obs = self.pygame.observe()
        return obs

    def play(self, max_try):
        """Simulate 1 episode of Single player game"""
        score = 0
        self.reset()
        for t in range(max_try):
            reward, done = self.pygame.run_game_loop()
            score += reward
            if done:
                break
        self.reset()
        return score, done, {'time': t}

    def close(self):
        """close the pygame object and window"""
        self.pygame.close()
        del self.pygame
