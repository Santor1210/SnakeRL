from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from snakeRLEnv import CustomEnv

model = DQN.load('best_model_17200000')

env = DummyVecEnv([lambda: CustomEnv(mode='human')])
env = VecFrameStack(env, 2, channels_order='last')

while True:
    state = env.reset()
    done = False
    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, reward, done, info = env.step(action)
        #state, reward, done, info = env.step(env.action_space.sample())
        env.render()
