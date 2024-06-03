import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from typing import Callable
from snakeRLEnv import CustomEnv

CHECKPOINT_DIR = './train'
LOG_DIR = './logs'


def linear_schedule_with_min(initial_value: float, min_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule with a minimum value.

    :param initial_value: Initial learning rate.
    :param min_value: Minimum learning rate to maintain.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        current_lr = progress_remaining * initial_value
        return max(current_lr, min_value)

    return func


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1, last_checkpoint=0, save_dir=None):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path_base = save_path
        self.last_checkpoint = last_checkpoint
        self.save_dir = save_dir

    def _init_callback(self):
        self.save_path = os.path.join(self.save_path_base, f'{self.save_dir}')
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        calls = self.n_calls + self.last_checkpoint
        if calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'best_model_{calls}')
            self.model.save(model_path)


callback = TrainAndLoggingCallback(check_freq=200000, save_path=CHECKPOINT_DIR, last_checkpoint=15800000, save_dir='')
model = DQN.load('best_model_17200000', tensorboard_log=LOG_DIR)
print(model.learning_rate)

#model.learning_rate = 0.000001
#model.batch_size = 128
#model.exploration_rate = 0.0001
#model.train_freq = TrainFreq(frequency=16, unit=TrainFrequencyUnit.STEP)
#model.gradient_steps = 8
#model.target_update_interval = 500

env = DummyVecEnv([lambda: CustomEnv()])  # Wrap your custom environment with DummyVecEnv
env = VecFrameStack(env, 2, channels_order='last')
model.set_env(env)
model.learn(total_timesteps=22400000, reset_num_timesteps=False, callback=callback)
